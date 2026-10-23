"""Reference pronunciation with Kokoro TTS (local, 24 kHz)."""
import hashlib
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from .registers import REGISTERS

CACHE = Path(__file__).resolve().parent.parent / "data" / "tts"
CACHE.mkdir(parents=True, exist_ok=True)
_pipes = {}


def _pipe(lang_code: str):
    if lang_code not in _pipes:
        from kokoro import KPipeline

        _pipes[lang_code] = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")
    return _pipes[lang_code]


def pick_voice(register: str, accent: str, gender: str, override: str | None = None) -> str:
    if override:
        return override
    return REGISTERS[register]["voices"][accent][gender]


def synth(text: str, register: str, accent: str = "us", gender: str = "female", voice: str | None = None,
          speed: float | None = None) -> tuple[str, Path]:
    voice = pick_voice(register, accent, gender, voice)
    speed = speed or REGISTERS[register]["speed"]
    key = hashlib.sha1(f"{text}|{voice}|{speed}".encode()).hexdigest()[:16]
    path = CACHE / f"{key}.wav"
    if not path.exists():
        pipe = _pipe(voice[0])  # 'a' = American, 'b' = British
        chunks = [np.asarray(audio) for _, _, audio in pipe(text, voice=voice, speed=speed) if audio is not None]
        audio = np.concatenate(chunks) if chunks else np.zeros(2400, dtype=np.float32)
        sf.write(path, audio, 24000)
    return key, path


def path_for(key: str) -> Path | None:
    p = CACHE / f"{os.path.basename(key)}.wav"
    return p if p.exists() else None
