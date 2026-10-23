"""Import your own movies, shows, podcasts or recordings (a local file or a URL) as reference clips.

Subtitles (.srt next to the file, or embedded/downloaded English subs) give the lines; without
them Whisper transcribes the whole file. Each line becomes a candidate that goes through the
same vetting as the public corpora (process.py).
"""
import re
import subprocess
from pathlib import Path

import numpy as np

from ..locks import MODEL_LOCK
from . import db

SR = 24000
MEDIA_DIR = db.LIB_DIR / "media_src"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "media"


def download_url(url: str, log) -> tuple[Path, Path | None, str]:
    import yt_dlp

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestaudio/best", "outtmpl": str(MEDIA_DIR / "%(title).80s [%(id)s].%(ext)s"),
        "writesubtitles": True, "subtitleslangs": ["en", "en-US", "en-GB"], "subtitlesformat": "srt/vtt/best",
        "noplaylist": True, "quiet": True, "no_warnings": True,
        "postprocessors": [{"key": "FFmpegSubtitlesConvertor", "format": "srt"}],
    }
    log("Downloading…")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
    subs = sorted(path.parent.glob(path.stem + "*.srt"))
    return path, (subs[0] if subs else None), info.get("title") or path.stem


def find_subs(path: Path) -> Path | None:
    for pat in (f"{path.stem}.en.srt", f"{path.stem}.eng.srt", f"{path.stem}.srt", f"{path.stem}*.srt"):
        hits = sorted(path.parent.glob(pat))
        if hits:
            return hits[0]
    return None


def extract_embedded_subs(path: Path) -> Path | None:
    out = MEDIA_DIR / f"{slug(path.stem)}.embedded.srt"
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-map", "0:s:m:language:eng?", "-map", "0:s:0?",
                        "-frames:s", "999999", str(out)], capture_output=True)
    return out if r.returncode == 0 and out.exists() and out.stat().st_size > 200 else None


def load_audio(path: Path) -> np.ndarray:
    p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1"],
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.int16)


def _ts(t: str) -> float:
    h, m, s = t.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_srt(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    cues = []
    for block in re.split(r"\n\s*\n", raw.replace("\r", "")):
        m = re.search(r"(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)", block)
        if not m:
            continue
        lines = [l for l in block[m.end():].strip().split("\n") if l.strip()]
        if sum(l.lstrip().startswith("-") for l in lines) >= 2:
            continue  # two speakers in one cue
        text = " ".join(lines)
        text = re.sub(r"<[^>]+>|\{[^}]+\}|\[[^\]]*\]|\([A-Z ]+\)|♪[^♪]*♪?|^-\s*", "", text).strip()
        text = re.sub(r"^[A-Z][A-Z .']+:\s*", "", text)  # "JOHN: hello"
        if re.search(r"[A-Za-z]", text):
            cues.append({"start": _ts(m.group(1)), "end": _ts(m.group(2)), "text": text})
    return cues


def merge_units(cues: list[dict], max_len: float = 12.0) -> list[dict]:
    """Join subtitle cues that continue one sentence; break at sentence ends or long gaps."""
    units, cur = [], None
    for c in cues:
        if cur and c["start"] - cur["end"] < 0.5 and not re.search(r"[.?!]['\"]?$", cur["text"]) \
                and c["end"] - cur["start"] <= max_len:
            cur["end"], cur["text"] = c["end"], cur["text"] + " " + c["text"]
        else:
            if cur:
                units.append(cur)
            cur = dict(c)
    if cur:
        units.append(cur)
    return units


def whisper_units(audio16: np.ndarray, log) -> list[dict]:
    from ..audio_asr import transcribe

    chunk = 16000 * 600
    words = []
    for i in range(0, len(audio16), chunk):
        log(f"Transcribing {i / 16000 / 60:.0f}–{min(len(audio16), i + chunk) / 16000 / 60:.0f} min…")
        with MODEL_LOCK:
            r = transcribe(audio16[i:i + chunk])
        words += [{**w, "start": w["start"] + i / 16000, "end": w["end"] + i / 16000} for w in r["words"]]
    units, cur = [], []
    for w in words:
        if cur and (w["start"] - cur[-1]["end"] > 0.6 or w["end"] - cur[0]["start"] > 12):
            units.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.?!]$", w["text"]):
            units.append(cur)
            cur = []
    if cur:
        units.append(cur)
    return [{"start": u[0]["start"], "end": u[-1]["end"], "text": " ".join(w["text"] for w in u)} for u in units]


def candidates(src: str, title: str | None, log):
    if re.match(r"https?://", src):
        path, subs, auto_title = download_url(src, log)
    else:
        path = Path(src).expanduser()
        if not path.exists():
            raise FileNotFoundError(src)
        subs, auto_title = find_subs(path), path.stem
    title = title or re.sub(r"\s*\[[^\]]+\]$", "", auto_title)
    log("Reading audio…")
    audio = load_audio(path)
    if subs is None and not re.match(r"https?://", src):
        subs = extract_embedded_subs(path)
    if subs:
        log(f"Using subtitles: {subs.name}")
        units = merge_units(parse_srt(subs))
    else:
        from scipy.signal import resample_poly

        log("No subtitles found, transcribing with Whisper…")
        a16 = resample_poly(audio.astype(np.float32) / 32768, 2, 3).astype(np.float32)
        units = whisper_units(a16, log)
    log(f"{len(units)} lines found.")
    key = slug(title)
    for i, u in enumerate(units):
        a, b = max(0, int((u["start"] - 0.3) * SR)), min(len(audio), int((u["end"] + 0.3) * SR))
        prev = [x["text"] for x in units[max(0, i - 2):i]]
        yield {
            "id": f"media-{key}-{int(u['start'] * 100)}", "source": f"media:{title}", "source_label": title,
            "speaker": None, "accent": None, "orig_text": u["text"], "context": " / ".join(prev),
            "audio": audio[a:b].astype(np.float32) / 32768, "sr": SR, "lenient": True,
        }
