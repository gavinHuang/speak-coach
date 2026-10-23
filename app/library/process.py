"""Turn a candidate utterance into a vetted, measured, labelled library clip.

Every candidate is re-transcribed with Whisper (gives casing/punctuation and catches wrong
dataset transcripts), force-aligned with the phoneme model (a low native-speaker score means
noise, music, laugh track or overlapping talk), and measured for pace and pitch.
"""
import re

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .. import prosody
from ..analysis import scorer, tokenize
from ..audio_asr import FILLERS, transcribe, trim_silence
from ..locks import MODEL_LOCK
from . import db

MIN_DUR, MAX_DUR = 1.2, 15.0
MIN_WORDS = 3


def norm_words(text: str) -> list[str]:
    text = re.sub(r"<[^>]*>|\[[^\]]*\]|\([^)]*\)", " ", text or "")
    text = text.replace("\x92", "'").replace("’", "'").lower()
    text = re.sub(r"[^a-z0-9 ]", "", text.replace("'", ""))
    return [w for w in text.split() if w not in FILLERS]


def wer(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 0.0
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / len(ref)


def _resample(audio: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return audio.astype(np.float32)
    g = np.gcd(sr, target)
    return resample_poly(audio, target // g, sr // g).astype(np.float32)


def process(c: dict) -> tuple[bool, str]:
    """c: id, source, source_label, speaker, gender, accent, orig_text, context, emotion, style, audio, sr."""
    if db.seen(c["id"]):
        return False, "seen"
    audio, sr = c["audio"], c["sr"]
    if audio.ndim > 1:
        audio = audio.mean(1)
    dur = len(audio) / sr
    if not (MIN_DUR <= dur <= MAX_DUR + 2):
        db.mark_rejected(c["id"], "duration")
        return False, "duration"
    a16 = _resample(audio, sr, 16000)

    with MODEL_LOCK:
        asr = transcribe(a16)
    words = asr["words"]
    lenient = c.get("lenient", False)
    if len(words) < MIN_WORDS:
        # Whisper sometimes hears nothing under film music; trust the subtitle and let the
        # phoneme alignment below decide whether the audio really matches it.
        if not (lenient and len(norm_words(c.get("orig_text"))) >= MIN_WORDS):
            db.mark_rejected(c["id"], "too few words")
            return False, "too few words"
        text, w_err = c["orig_text"], None
        if not (MIN_DUR <= dur <= MAX_DUR):
            db.mark_rejected(c["id"], "duration")
            return False, "duration"
    else:
        # Crop to the speech itself (subtitle and dataset timings are often loose).
        t0, t1 = max(0.0, words[0]["start"] - 0.15), min(dur, words[-1]["end"] + 0.25)
        audio = audio[int(t0 * sr): int(t1 * sr)]
        a16 = trim_silence(a16[int(t0 * 16000): int(t1 * 16000)])
        dur = len(audio) / sr
        if not (MIN_DUR <= dur <= MAX_DUR):
            db.mark_rejected(c["id"], "duration")
            return False, "duration"
        text = asr["text"].strip()
        w_err = wer(norm_words(c.get("orig_text")), norm_words(text)) if c.get("orig_text") else None
    if w_err is not None and w_err > 0.35:
        db.mark_rejected(c["id"], f"transcript mismatch {w_err:.2f}")
        return False, "transcript mismatch"

    toks = tokenize(text)
    with MODEL_LOCK:
        pron, _ = scorer().score(a16, [t["clean"] for t in toks], "en-us")
    scored = [p for p, t in zip(pron, toks) if p["score"] is not None and t["clean"] not in FILLERS]
    if not scored:
        db.mark_rejected(c["id"], "alignment failed")
        return False, "alignment failed"
    pron_score = float(np.mean([p["score"] for p in scored]))
    tr = prosody.tracks(a16)
    stats = prosody.pitch_stats(tr, 0, len(a16) / 16000)
    db_vals = tr["db"][np.isfinite(tr["db"])]
    snr = float(np.percentile(db_vals, 95) - np.percentile(db_vals, 10)) if len(db_vals) else 0.0
    # Native speakers in clean audio score ~85+; much lower means the audio is not clean speech.
    if pron_score < (62 if lenient else 68):
        db.mark_rejected(c["id"], f"unclear audio {pron_score:.0f}")
        return False, "unclear audio"
    if snr < (11 if lenient else 15):
        db.mark_rejected(c["id"], f"noisy {snr:.0f}dB")
        return False, "noisy"
    last = scored[-1]
    contour, _ = prosody.final_contour(tr, last["start"], last["end"]) if last["start"] is not None else ("unclear", None)
    n_words = sum(t["clean"] not in FILLERS for t in toks)
    gender = c.get("gender") or (("male" if stats["median_hz"] < 165 else "female") if stats["median_hz"] else None)
    quality = int(np.clip(0.6 * (pron_score - 60) * 2.5 + 0.4 * min(100, (snr - 10) * 4) - 40 * (w_err or 0), 0, 100))

    out_sr = min(sr, 24000)
    rel = f"{re.sub(r'[^a-z0-9_-]', '_', c['source'].lower())}/{c['id']}.flac"
    path = db.CLIPS_DIR / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, _resample(audio, sr, out_sr), out_sr, format="FLAC")
    db.insert_clip({
        **{k: c.get(k) for k in ("id", "source", "source_label", "speaker", "accent", "orig_text", "context", "emotion", "style")},
        "gender": gender, "text": text, "duration": round(dur, 2),
        "wpm": round(n_words / max(0.5, dur) * 60), "pitch_sd": stats["pitch_sd_st"], "pitch_range": stats["pitch_range_st"],
        "median_hz": stats["median_hz"], "final_contour": contour, "quality": quality, "pron_score": round(pron_score),
        "wer": None if w_err is None else round(w_err, 2), "snr_db": round(snr, 1), "path": rel,
    })
    return True, "ok"


TONES = ("neutral, warm, playful, excited, enthusiastic, annoyed, angry, sad, anxious, apologetic, persuasive, confident, "
         "hesitant, sarcastic, comforting, affectionate, flirty, curious, surprised, formal, polite, frustrated, tender, joking")
ACTS = ("question, request, suggestion, opinion, explanation, agreement, disagreement, apology, thanks, greeting, "
        "complaint, compliment, reassurance, storytelling, joke, refusal, offer, instruction, update, reaction, small-talk")


INTIMATE_TONES = {"affectionate", "tender", "flirty", "comforting"}


def label_pending(batch: int = 15, max_batches: int | None = None, progress=None) -> int:
    """Ask the local LLM for register / tone / speech act / usefulness of unlabelled clips."""
    from .. import llm

    done = n = 0
    while max_batches is None or n < max_batches:
        rows = db.unlabeled(batch)
        if not rows:
            break
        n += 1
        lines = []
        for i, r in enumerate(rows, 1):
            meta = "; ".join(x for x in [r["source_label"], r["speaker"] and f"speaker {r['speaker']}",
                                         r["emotion"] and f"emotion {r['emotion']}", r["style"] and f"style {r['style']}"] if x)
            ctx = f' (previous: {r["context"][-160:]})' if r["context"] else ""
            lines.append(f'{i}. [{meta}]{ctx} "{r["text"]}"')
        with MODEL_LOCK:
            res = llm.chat_json(
                "You are an expert in English sociolinguistics labelling real spoken lines for language learners.",
                "Label each numbered line of real spoken English.\n"
                "register: work (colleagues, clients, meetings, presentations, service, formal), casual (friends, "
                "family banter, everyday chat, joking around), intimate (romantic, flirting, affectionate, tender, heartfelt "
                "confessions, consoling or reassuring someone you love, saying sorry to a partner, talk between partners, lovers, "
                "or very close family/friends about feelings).\n"
                f"tone: one word from: {TONES}.\nact: one from: {ACTS}.\n"
                "usefulness: 1-5, how good this is as a phrase for a learner to shadow (natural, self-contained, reusable "
                "outside the show/meeting; 1 = fragment, names-only, or meaningless out of context).\n\n" + "\n".join(lines) +
                '\n\nReturn JSON: [{"i": 1, "register": "...", "tone": "...", "act": "...", "usefulness": 3}, ...]',
                max_tokens=80 * len(rows) + 100, temp=0.1,
            )
        got = {}
        if isinstance(res, list):
            for x in res:
                if isinstance(x, dict) and isinstance(x.get("i"), int) and 1 <= x["i"] <= len(rows):
                    got[x["i"]] = x
        for i, r in enumerate(rows, 1):
            x = got.get(i, {})
            reg = x.get("register") if x.get("register") in ("work", "casual", "intimate") else "casual"
            tone = str(x.get("tone") or "neutral").lower().split()[0][:20]
            if reg == "casual" and tone in INTIMATE_TONES:
                reg = "intimate"
            act = str(x.get("act") or "").lower().strip()[:20] or None
            try:
                use = int(x.get("usefulness", 2))
            except (TypeError, ValueError):
                use = 2
            db.set_labels(r["id"], reg, tone, act, max(1, min(5, use)))
            done += 1
        if progress:
            progress(done)
    return done
