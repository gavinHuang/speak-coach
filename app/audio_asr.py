"""Audio I/O and speech recognition (Whisper on Apple Silicon via MLX)."""
import os
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
# Priming Whisper with disfluent text makes it keep "um/uh" instead of silently cleaning them up.
FILLER_PROMPT = "Umm, so, uh, I was like... you know, I mean, hmm, er, okay."
FILLERS = {"um", "umm", "uh", "uhh", "er", "erm", "hmm", "mm", "ah", "eh"}
DISCOURSE_MARKERS = ["you know", "i mean", "like", "kind of", "sort of", "basically", "actually"]


def to_wav16k(src: str, dst: str) -> np.ndarray:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "16000", dst], check=True
    )
    return load16k(dst)


def load16k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    if sr != 16000:
        audio = resample_poly(audio, 16000, sr).astype(np.float32)
    return audio


def trim_silence(audio: np.ndarray, thresh_db: float = -40) -> np.ndarray:
    """Cut leading/trailing silence (keeps 0.15 s margin)."""
    frame = 320
    n = len(audio) // frame
    if n == 0:
        return audio
    rms = np.sqrt((audio[: n * frame].reshape(n, frame) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms / (rms.max() + 1e-12))
    idx = np.where(db > thresh_db)[0]
    if len(idx) == 0:
        return audio
    a = max(0, (idx[0] - 8) * frame)
    b = min(len(audio), (idx[-1] + 8) * frame)
    return audio[a:b]


def transcribe(audio: np.ndarray) -> dict:
    import mlx_whisper

    r = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=WHISPER_MODEL,
        word_timestamps=True,
        initial_prompt=FILLER_PROMPT,
        language="en",
        condition_on_previous_text=False,
        temperature=0.0,
    )
    words = []
    for seg in r.get("segments", []):
        for w in seg.get("words", []):
            text = w["word"].strip()
            if text:
                words.append(
                    {
                        "text": text,
                        "start": float(w["start"]),
                        "end": float(w["end"]),
                        "prob": float(w.get("probability", 1.0)),
                    }
                )
    return {"text": r.get("text", "").strip(), "words": words}
