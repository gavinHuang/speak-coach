"""Prosody: pitch (intonation), loudness, timing — measured with Praat (parselmouth)."""
import numpy as np
import parselmouth
from scipy.signal import medfilt

WH_WORDS = {"what", "where", "when", "why", "who", "whom", "whose", "which", "how"}


def tracks(audio: np.ndarray, sr: int = 16000) -> dict:
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=65, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    t = pitch.xs()
    inten = snd.to_intensity(minimum_pitch=75, time_step=0.01)
    iv = np.interp(t, inten.xs(), inten.values[0])
    voiced = f0 > 0
    med = float(np.median(f0[voiced])) if voiced.sum() > 5 else 0.0
    st = np.full_like(f0, np.nan)
    if med:
        st[voiced] = 12 * np.log2(f0[voiced] / med)
        st[np.abs(st) > 12] = np.nan  # octave errors
        good = ~np.isnan(st)
        if good.sum() > 5:
            st[good] = medfilt(st[good], 5)
    return {"t": t, "f0": f0, "st": st, "median_hz": med, "db": iv}


def window(tr: dict, t0: float, t1: float):
    m = (tr["t"] >= t0) & (tr["t"] < t1)
    return tr["st"][m], tr["db"][m]


def pitch_stats(tr: dict, t0: float, t1: float) -> dict:
    st, db = window(tr, t0, t1)
    v = st[~np.isnan(st)]
    if len(v) < 10:
        return {"pitch_sd_st": None, "pitch_range_st": None, "median_hz": round(tr["median_hz"])}
    speech_db = db[db > np.percentile(db, 30)] if len(db) else db
    return {
        "pitch_sd_st": round(float(np.std(v)), 2),
        "pitch_range_st": round(float(np.percentile(v, 95) - np.percentile(v, 5)), 1),
        "median_hz": round(tr["median_hz"]),
        "loudness_sd_db": round(float(np.std(speech_db)), 1) if len(speech_db) else None,
    }


def final_contour(tr: dict, t0: float, t1: float) -> tuple[str, float | None]:
    """Direction of the pitch movement over the final word of a sentence."""
    st, _ = window(tr, t0, t1)
    v = st[~np.isnan(st)]
    if len(v) < 6:
        return "unclear", None
    half = len(v) // 2
    delta = float(np.mean(v[half:]) - np.mean(v[:half]))
    tail = float(v[-3:].mean() - v.min())
    if delta > 1.5 or (tail > 3 and delta > 0.5):
        return "rising", round(delta, 1)
    if delta < -1.5:
        return "falling", round(delta, 1)
    return "level", round(delta, 1)


def expected_final(sentence_words: list[str], punct: str) -> str:
    if punct == "?":
        first = sentence_words[0].lower() if sentence_words else ""
        return "falling" if first in WH_WORDS else "rising"
    return "falling"


def word_features(tr: dict, words: list[dict]) -> list[dict]:
    """Per-word mean pitch, peak loudness and length; used for emphasis/rhythm comparison."""
    feats = []
    for w in words:
        if w.get("start") is None:
            feats.append(None)
            continue
        st, db = window(tr, w["start"], w["end"])
        v = st[~np.isnan(st)]
        n = max(1, len(w.get("phones", [])))
        feats.append(
            {
                "pitch": float(np.percentile(v, 90)) if len(v) >= 3 else np.nan,
                "loud": float(db.max()) if len(db) else np.nan,
                "dur": (w["end"] - w["start"]) / n,
            }
        )
    return feats


def prominence(feats: list[dict | None]) -> np.ndarray:
    """Combined z-score of pitch, loudness and (per-phone) duration for each word."""
    keys = ["pitch", "loud", "dur"]
    mat = np.array([[f[k] if f else np.nan for k in keys] for f in feats], dtype=float)
    out = np.zeros(len(feats))
    for j in range(mat.shape[1]):
        col = mat[:, j]
        ok = ~np.isnan(col)
        if ok.sum() >= 2 and np.std(col[ok]) > 1e-6:
            z = np.zeros_like(col)
            z[ok] = (col[ok] - col[ok].mean()) / col[ok].std()
            out += z
    return out


def contour_for_plot(tr: dict, step: int = 2) -> dict:
    t = tr["t"][::step]
    st = tr["st"][::step]
    return {"t": [round(float(x), 3) for x in t], "st": [None if np.isnan(x) else round(float(x), 2) for x in st]}
