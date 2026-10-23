"""Public real-speech corpora the library can draw from. Downloads go to the Hugging Face cache."""
import csv
import fnmatch
import io
import random
import re
import subprocess
import tarfile
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

CATALOG = {
    "meld": {
        "label": "Friends (TV sitcom)", "repo": "ajyy/MELD_audio", "kind": "TV dialogue",
        "license": "MELD research dataset; show audio © Warner Bros. Personal study use.",
        "about": "13.7k lines from 10 seasons of Friends with emotion labels: banter, dating, arguments, comfort.",
        "size": "1.5 GB", "accent": "American",
    },
    "ami": {
        "label": "AMI business meetings", "repo": "edinburghcstr/ami", "kind": "meetings", "license": "CC BY 4.0",
        "about": "Real design-team meetings (headset mics): proposing, disagreeing, deciding, small talk.",
        "size": "~300 MB per shard (42)", "accent": "British / European / international",
    },
    "earnings22": {
        "label": "Earnings calls", "repo": "distil-whisper/earnings22", "kind": "business calls", "license": "CC BY-SA 4.0",
        "about": "Executives and analysts on real company earnings calls: presenting results, Q&A.",
        "size": "~600 MB per shard (38)", "accent": "international",
    },
    "voxpopuli": {
        "label": "European Parliament speeches", "repo": "facebook/voxpopuli", "kind": "speeches", "license": "CC0",
        "about": "Formal public speaking by MEPs, many of them non-native but fluent.",
        "size": "0.9 GB (test split)", "accent": "European",
    },
    "expresso": {
        "label": "Expresso expressive speech", "repo": "ylacombe/expresso", "kind": "acted styles", "license": "CC BY-NC 4.0",
        "about": "The same kinds of lines in many tones: whisper, happy, sad, laughing, confused, enunciated…",
        "size": "~500 MB per shard (12)", "accent": "American",
    },
    "edacc": {
        "label": "EdAcc accented conversations", "repo": "edinburghcstr/edacc", "kind": "conversations", "license": "CC BY-SA 4.0",
        "about": "Relaxed video-call conversations between friends with 40+ accents of English.",
        "size": "~350 MB per shard (10)", "accent": "40+ accents",
    },
}

EXPRESSO_STYLE_HINT = {"whisper": "intimate", "sad": "intimate", "happy": "casual", "laughing": "casual",
                       "confused": "casual", "enunciated": "work", "default": "", "narration": "work"}


def _files(repo: str, pattern: str) -> list[str]:
    info = HfApi().dataset_info(repo)
    return sorted(s.rfilename for s in info.siblings if fnmatch.fnmatch(s.rfilename, pattern))


def _decode(b: bytes):
    try:
        a, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
        return a, sr
    except Exception:
        p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
                           input=b, capture_output=True, check=True)
        return np.frombuffer(p.stdout, dtype=np.float32), 16000


def _parquet_rows(repo: str, pattern: str, seed: int):
    import pyarrow.parquet as pq

    files = _files(repo, pattern)
    random.Random(seed).shuffle(files)
    for f in files:
        path = hf_hub_download(repo, f, repo_type="dataset")
        pf = pq.ParquetFile(path)
        for g in range(pf.num_row_groups):
            yield f, pf.read_row_group(g).to_pylist()


def _fix(t: str) -> str:
    for a, b in (("\x91", "'"), ("\x92", "'"), ("\x93", '"'), ("\x94", '"'), ("\x85", "..."), ("\x96", "-"), ("\x97", "-"), ("\xa0", " ")):
        t = (t or "").replace(a, b)
    return t.strip()


def _ctx(prev: list[tuple[str, str]]) -> str:
    return " / ".join(f"{s}: {t}" for s, t in prev[-2:] if t)


def iter_meld(seed: int):
    rows = {}
    for split in ("train", "dev", "test"):
        p = hf_hub_download("ajyy/MELD_audio", f"{split}.csv", repo_type="dataset")
        with open(p, encoding="utf-8", errors="replace", newline="") as f:
            for r in csv.DictReader(f):
                rows[(split, int(r["Dialogue_ID"]), int(r["Utterance_ID"]))] = r
    for split in ("dev", "test", "train"):
        tar_path = hf_hub_download("ajyy/MELD_audio", f"archive/{split}.tar.gz", repo_type="dataset")
        with tarfile.open(tar_path, "r:gz") as tf:
            for m in tf:
                mm = re.search(r"dia(\d+)_utt(\d+)\.(wav|flac|mp3)$", m.name)
                if not mm or not m.isfile():
                    continue
                d, u = int(mm.group(1)), int(mm.group(2))
                r = rows.get((split, d, u))
                if not r:
                    continue
                prev = [(rows[(split, d, k)]["Speaker"], _fix(rows[(split, d, k)]["Utterance"]))
                        for k in range(max(0, u - 2), u) if (split, d, k) in rows]
                yield {
                    "id": f"meld-{split}-{d}-{u}", "source": "meld",
                    "source_label": f"Friends S{int(r['Season']):02d}E{int(r['Episode']):02d}",
                    "speaker": r["Speaker"], "accent": "American", "orig_text": _fix(r["Utterance"]),
                    "context": _ctx(prev), "emotion": r["Emotion"], "group": f"{split}-{d}",
                    "_bytes": tf.extractfile(m).read(),
                }


def iter_ami(seed: int):
    for f, rows in _parquet_rows("edinburghcstr/ami", "ihm/train-*.parquet", seed):
        rows.sort(key=lambda r: (r["meeting_id"], r["begin_time"]))
        last = defaultdict(list)
        for r in rows:
            m = r["meeting_id"]
            text = r["text"].strip()
            ctx = _ctx(last[m])
            last[m].append((r["speaker_id"], text.lower()))
            yield {
                "id": f"ami-{r['audio_id']}", "source": "ami", "source_label": f"AMI meeting {m}",
                "speaker": r["speaker_id"], "gender": {"M": "male", "F": "female"}.get(r["speaker_id"][:1]),
                "accent": "various (meeting)", "orig_text": text, "context": ctx, "group": m,
                "_bytes": r["audio"]["bytes"],
            }


def iter_earnings22(seed: int):
    for f, rows in _parquet_rows("distil-whisper/earnings22", "chunked/test-*.parquet", seed):
        rows.sort(key=lambda r: (r["file_id"], r["start_ts"]))
        last = defaultdict(list)
        for r in rows:
            text = _fix(r["transcription"])
            ctx = _ctx(last[r["file_id"]])
            last[r["file_id"]].append(("Speaker", text))
            yield {
                "id": f"earn-{r['file_id']}-{r['segment_id']}", "source": "earnings22",
                "source_label": f"Earnings call {r['file_id']}", "speaker": None, "accent": "international",
                "orig_text": text, "context": ctx, "group": r["file_id"], "_bytes": r["audio"]["bytes"],
            }


def iter_voxpopuli(seed: int):
    for f, rows in _parquet_rows("facebook/voxpopuli", "en/test-*.parquet", seed):
        for r in rows:
            text = r.get("raw_text") or r.get("normalized_text") or ""
            yield {
                "id": f"vox-{r.get('audio_id')}", "source": "voxpopuli", "source_label": "European Parliament",
                "speaker": r.get("speaker_id"), "gender": r.get("gender") or None,
                "accent": r.get("accent") if r.get("accent") not in (None, "", "None") else "European",
                "orig_text": _fix(text), "context": "", "group": r.get("speaker_id"), "_bytes": r["audio"]["bytes"],
            }


def iter_expresso(seed: int):
    for f, rows in _parquet_rows("ylacombe/expresso", "read/train-*.parquet", seed):
        random.Random(seed).shuffle(rows)
        for r in rows:
            yield {
                "id": f"expr-{r['id']}", "source": "expresso", "source_label": f"Expresso ({r['style']})",
                "speaker": r["speaker_id"], "accent": "American", "orig_text": _fix(r["text"]), "context": "",
                "style": r["style"], "group": f"{r['speaker_id']}-{r['style']}", "_bytes": r["audio"]["bytes"],
            }


def iter_edacc(seed: int):
    for f, rows in _parquet_rows("edinburghcstr/edacc", "data/*.parquet", seed):
        last = []
        for i, r in enumerate(rows):
            conv = r["speaker"].rsplit("-", 1)[0]
            if last and last[-1][0] != conv:
                last = []
            text = re.sub(r"<[^>]+>", "", r["text"]).strip()
            ctx = _ctx([(s, t) for _, s, t in last])
            last.append((conv, r["speaker"][-1], text.lower()))
            yield {
                "id": f"edacc-{f.split('/')[-1][:10]}-{i}", "source": "edacc", "source_label": f"EdAcc conversation {conv}",
                "speaker": r["speaker"], "gender": r.get("gender"), "accent": r.get("accent") or r.get("l1"),
                "orig_text": text, "context": ctx, "group": conv, "_bytes": r["audio"]["bytes"],
            }


ITERATORS = {"meld": iter_meld, "ami": iter_ami, "earnings22": iter_earnings22, "voxpopuli": iter_voxpopuli,
             "expresso": iter_expresso, "edacc": iter_edacc}


def candidates(source: str, max_new: int, seed: int = 0):
    """Yield decoded candidates with cheap pre-filters and per-speaker / per-conversation caps for variety."""
    from . import db

    per_group, per_speaker = Counter(), Counter()
    cap_g, cap_s = max(8, max_new // 25), max(15, max_new // 12)
    for c in ITERATORS[source](seed):
        n_words = len(re.findall(r"[A-Za-z']+", re.sub(r"<[^>]*>", "", c["orig_text"])))
        if n_words < 4 or n_words > 45:
            continue
        g, s = c.get("group"), c.get("speaker")
        if per_group[g] >= cap_g or (s and per_speaker[s] >= cap_s):
            continue
        if db.seen(c["id"]):
            continue
        try:
            c["audio"], c["sr"] = _decode(c.pop("_bytes"))
        except Exception:
            continue
        per_group[g] += 1
        if s:
            per_speaker[s] += 1
        yield c
