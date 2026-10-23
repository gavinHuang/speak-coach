"""Runs every analysis on one recording and assembles the report."""
import difflib
import re
from collections import Counter

import numpy as np

from . import prosody
from .audio_asr import DISCOURSE_MARKERS, FILLERS, transcribe, trim_silence
from .phonemes import PhonemeScorer, is_vowel
from .registers import PHONE_TIPS, REGISTERS

_scorer: PhonemeScorer | None = None


def scorer() -> PhonemeScorer:
    global _scorer
    if _scorer is None:
        _scorer = PhonemeScorer()
    return _scorer


def tokenize(text: str) -> list[dict]:
    out = []
    for tok in re.sub(r"[-—–/]", " ", text).split():
        clean = re.sub(r"[^A-Za-z0-9']", "", tok).strip("'").lower()
        m = re.search(r"([.?!])[\"')\]]*$", tok)
        if not clean:
            if out and m:
                out[-1]["punct"] = m.group(1)
            continue
        out.append({"text": tok, "clean": clean, "punct": m.group(1) if m else "", "comma": bool(re.search(r"[,;:]$", tok))})
    return out


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3 or a[ok].std() < 1e-6 or b[ok].std() < 1e-6:
        return None
    return round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 2)


def _clip(x):
    return int(max(0, min(100, round(x))))


def _stress_check(w, tr):
    vowels = [p for p in w["phones"] if is_vowel(p["p"])]
    if len(vowels) < 2 or not any(v["stress"] == 1 for v in vowels):
        return
    feats = []
    for v in vowels:
        st, db = prosody.window(tr, v["t0"], v["t1"])
        vv = st[~np.isnan(st)]
        feats.append([vv.mean() if len(vv) else np.nan, db.max() if len(db) else np.nan, v["t1"] - v["t0"]])
    mat = np.array(feats, float)
    prom = np.zeros(len(vowels))
    for j in range(3):
        col = mat[:, j]
        ok = ~np.isnan(col)
        if ok.sum() >= 2 and col[ok].std() > 1e-6:
            z = np.zeros_like(col)
            z[ok] = (col[ok] - col[ok].mean()) / col[ok].std()
            prom += z
    exp_i = next(i for i, v in enumerate(vowels) if v["stress"] == 1)
    got_i = int(prom.argmax())
    w["stress_expected"], w["stress_heard"] = exp_i, got_i
    if got_i != exp_i and prom[got_i] - prom[exp_i] > 2.0:
        w["stress_issue"] = f"stress sounds like it's on syllable {got_i + 1}; it should be on syllable {exp_i + 1}"


def _silences(tr, t0, t1, min_len=0.3):
    db, t = tr["db"], tr["t"]
    m = (t >= t0) & (t <= t1)
    if m.sum() < 10:
        return []
    thr = np.percentile(db[m], 90) - 25
    quiet = (db < thr) & m
    out, start = [], None
    for ti, q in zip(t, quiet):
        if q and start is None:
            start = ti
        elif not q and start is not None:
            if ti - start >= min_len:
                out.append((float(start), float(ti)))
            start = None
    return out


def analyze(audio, mode, register, accent="us", target_text="", prompt="", ref_audio=None) -> dict:
    reg = REGISTERS[register]
    lang = "en-us" if accent == "us" else "en-gb"
    audio = trim_silence(audio)
    asr = transcribe(audio)
    hyp_toks = []
    for w in asr["words"]:
        t = tokenize(w["text"])
        if t:
            hyp_toks.append({**t[0], "start": w["start"], "end": w["end"], "prob": w["prob"]})

    read = mode == "read" and target_text.strip()
    toks = tokenize(target_text) if read else hyp_toks
    clean = [t["clean"] for t in toks]
    sc = scorer()
    pron, _ = sc.score(audio, clean, lang) if clean else ([], None)
    ipa = sc.ipa(clean, lang) if clean else []
    tr = prosody.tracks(audio)

    words = []
    for t, p, i in zip(toks, pron, ipa):
        w = {
            "text": t["text"], "clean": t["clean"], "punct": t["punct"], "ipa": i,
            "phones": p["phones"], "start": p["start"], "end": p["end"], "score": p["score"], "heard": p["heard"],
            "filler": t["clean"] in FILLERS,
        }
        if w["filler"]:
            w["score"] = None
        elif w["phones"]:
            _stress_check(w, tr)
        words.append(w)

    # ---- reading accuracy: does the recogniser hear the words you were meant to say? ----
    accuracy = None
    if read:
        tgt = [w["clean"] for w in words]
        hyp = [t["clean"] for t in hyp_toks if t["clean"] not in FILLERS]
        sm = difflib.SequenceMatcher(a=tgt, b=hyp, autojunk=False)
        errs, extras = 0, []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            errs += max(i2 - i1, j2 - j1)
            if tag == "insert":
                extras += hyp[j1:j2]
            for k in range(i1, i2):
                if tag == "delete":
                    words[k]["asr"] = "(missing)"
                    if words[k]["score"] is not None:
                        words[k]["score"] = min(words[k]["score"], 40)
                else:
                    words[k]["asr"] = " ".join(hyp[j1:j2])
                    if words[k]["score"] is not None:
                        words[k]["score"] = min(words[k]["score"], 65)
        wer = errs / max(1, len(tgt))
        accuracy = {"wer": round(wer, 2), "extra_words": extras, "score": _clip(100 * (1 - wer))}

    # ---- fluency ----
    spoken = [t for t in hyp_toks]
    fl = {}
    if spoken:
        t0, t1 = spoken[0]["start"], spoken[-1]["end"]
        dur = max(0.5, t1 - t0)
        sil = _silences(tr, t0, t1)
        mid = 0
        for a, _ in sil:
            prev = [t for t in spoken if (t["start"] + t["end"]) / 2 < a]
            if prev and not (prev[-1]["punct"] or prev[-1]["comma"]):
                mid += 1
        n_fill = sum(t["clean"] in FILLERS for t in spoken)
        lower = " " + " ".join(t["clean"] for t in spoken) + " "
        markers = {m: lower.count(f" {m} ") for m in DISCOURSE_MARKERS if lower.count(f" {m} ")}
        reps = sum(1 for a, b in zip(spoken, spoken[1:]) if a["clean"] == b["clean"] and a["clean"] not in FILLERS)
        syl = sum(1 for w in words for p in w["phones"] if is_vowel(p["p"]))
        pause_time = sum(b - a for a, b in sil)
        n_words = sum(t["clean"] not in FILLERS for t in spoken)
        fl = {
            "duration_s": round(dur, 1),
            "wpm": round(n_words / dur * 60),
            "articulation_syl_per_s": round(syl / max(0.5, dur - pause_time), 2),
            "pauses": len(sil),
            "long_pauses": sum(1 for a, b in sil if b - a >= 1.0),
            "mid_phrase_pauses": mid,
            "pause_time_pct": round(100 * pause_time / dur),
            "mean_length_of_run_syl": round(syl / (len(sil) + 1), 1),
            "fillers": n_fill,
            "fillers_per_min": round(n_fill / dur * 60, 1),
            "discourse_markers": markers,
            "repetitions": reps,
        }

    # ---- intonation ----
    span = (words[0]["start"], words[-1]["end"]) if words and words[0]["start"] is not None else (0, len(audio) / 16000)
    into = prosody.pitch_stats(tr, *span)
    sentences, cur = [], []
    for w in words:
        if not w["filler"]:
            cur.append(w)
        if w["punct"] and cur:
            sentences.append((cur, w["punct"]))
            cur = []
    if cur:
        sentences.append((cur, "."))
    sent_out = []
    for sw, punct in sentences:
        last = sw[-1]
        if last["start"] is None:
            continue
        got, delta = prosody.final_contour(tr, last["start"], last["end"])
        exp = prosody.expected_final([w["clean"] for w in sw], punct)
        status = "ok" if got in (exp, "unclear") else ("warn" if got == "level" else "miss")
        if status == "miss" and exp == "rising":
            status = "warn"  # falling yes/no questions exist too; they just sound more insistent
        sent_out.append({"text": " ".join(w["text"] for w in sw), "expected": exp, "got": got, "delta_st": delta, "status": status})
    into["sentences"] = sent_out

    # ---- comparison with the reference voice ----
    ref = None
    if read and ref_audio is not None:
        ref_audio = trim_silence(ref_audio)
        ref_pron, _ = sc.score(ref_audio, clean, lang)
        ref_tr = prosody.tracks(ref_audio)
        uf, rf = prosody.word_features(tr, words), prosody.word_features(ref_tr, ref_pron)
        valid = [i for i in range(len(words)) if uf[i] and rf[i] and not words[i]["filler"]]
        up = prosody.prominence([uf[i] for i in valid])
        rp = prosody.prominence([rf[i] for i in valid])
        emph_corr = _pearson(up, rp)
        pitch_corr = _pearson([uf[i]["pitch"] for i in valid], [rf[i]["pitch"] for i in valid])
        k = max(1, len(valid) // 5)
        ref_top = set(np.argsort(-rp)[:k]) if len(valid) else set()
        usr_top = set(np.argsort(-up)[:k]) if len(valid) else set()
        missed = [words[valid[j]]["text"] for j in ref_top if up[j] < 0]
        extra = [words[valid[j]]["text"] for j in usr_top if rp[j] < 0]
        u_dur = (words[valid[-1]]["end"] - words[valid[0]]["start"]) if valid else None
        r_dur = (ref_pron[valid[-1]]["end"] - ref_pron[valid[0]]["start"]) if valid else None
        anchors_r, anchors_u = [], []
        for i in valid:
            anchors_r += [ref_pron[i]["start"], ref_pron[i]["end"]]
            anchors_u += [words[i]["start"], words[i]["end"]]
        rc = prosody.contour_for_plot(ref_tr)
        if anchors_r:
            ar = np.maximum.accumulate(np.array(anchors_r))
            au = np.maximum.accumulate(np.array(anchors_u))
            rt = np.array(rc["t"])
            keep = (rt >= ar[0]) & (rt <= ar[-1])
            rc = {"t": [round(float(x), 3) for x in np.interp(rt[keep], ar, au)],
                  "st": [s for s, kk in zip(rc["st"], keep) if kk]}
        ref = {
            "emphasis_match": emph_corr,
            "pitch_shape_match": pitch_corr,
            "pace_ratio": round(u_dur / r_dur, 2) if u_dur and r_dur else None,
            "missed_emphasis": missed,
            "extra_emphasis": extra,
            "ref_pitch_sd_st": prosody.pitch_stats(ref_tr, 0, len(ref_audio) / 16000)["pitch_sd_st"],
            "contour": rc,
            "words": [{"i": i, "text": words[i]["text"], "you": round(float(up[j]), 2), "ref": round(float(rp[j]), 2),
                       "ref_start": ref_pron[i]["start"], "ref_end": ref_pron[i]["end"]} for j, i in enumerate(valid)],
        }

    # ---- register checks ----
    checks = []
    if fl:
        lo, hi = reg["wpm"]
        checks.append({"metric": "Pace", "value": f'{fl["wpm"]} wpm', "target": f"{lo}–{hi} wpm", "ok": lo <= fl["wpm"] <= hi,
                       "msg": "" if lo <= fl["wpm"] <= hi else ("Slow down a little." if fl["wpm"] > hi else "Try a bit faster and smoother.")})
        checks.append({"metric": "Fillers", "value": f'{fl["fillers_per_min"]}/min', "target": f'≤ {reg["fillers_per_min"]}/min',
                       "ok": fl["fillers_per_min"] <= reg["fillers_per_min"], "msg": "Replace um/uh with a short silent pause."})
        checks.append({"metric": "Pauses inside phrases", "value": str(fl["mid_phrase_pauses"]), "target": "0–1",
                       "ok": fl["mid_phrase_pauses"] <= 1, "msg": "Pause at commas and full stops, not between words of one idea."})
    if into.get("pitch_sd_st") is not None:
        lo, hi = reg["pitch_sd"]
        sd = into["pitch_sd_st"]
        checks.append({"metric": "Pitch variation", "value": f"{sd} st", "target": f"{lo}–{hi} st", "ok": lo <= sd <= hi,
                       "msg": "" if lo <= sd <= hi else ("Sounds flat/monotone — move your voice up on key words." if sd < lo else "Very wide swings — calmer pitch suits this situation.")})

    # ---- scores ----
    scored = [w for w in words if w["score"] is not None]
    pron_score = _clip(np.average([w["score"] for w in scored], weights=[len(w["phones"]) for w in scored])) if scored else None
    flu = None
    if fl:
        lo, hi = reg["wpm"]
        f = 100.0
        f -= min(30, 6 * max(0, fl["fillers_per_min"] - 1))
        f -= min(25, 6 * fl["long_pauses"])
        f -= min(20, 5 * max(0, fl["mid_phrase_pauses"] - 1))
        f -= min(25, 0.6 * max(0, lo - fl["wpm"], fl["wpm"] - hi))
        f -= min(15, 4 * fl["repetitions"])
        if fl["mean_length_of_run_syl"] < 6:
            f -= 10
        flu = _clip(f)
    into_score = None
    if into.get("pitch_sd_st") is not None:
        lo, hi = reg["pitch_sd"]
        sd = into["pitch_sd_st"]
        range_score = 100 - (35 * (lo - sd) if sd < lo else 15 * (sd - hi) if sd > hi else 0)
        if sent_out:
            end_score = 100 * sum(1 if s["status"] == "ok" else 0.5 if s["status"] == "warn" else 0 for s in sent_out) / len(sent_out)
            base = 0.5 * range_score + 0.5 * end_score
        else:
            base = range_score
        if ref and ref["emphasis_match"] is not None:
            m = 100 * max(0.0, (ref["emphasis_match"] + (ref["pitch_shape_match"] or ref["emphasis_match"])) / 2)
            base = 0.4 * base + 0.6 * m
        into_score = _clip(base)
    scores = {"pronunciation": pron_score, "fluency": flu, "intonation": into_score}
    if accuracy:
        scores["accuracy"] = accuracy["score"]
    vals = [v for v in scores.values() if v is not None]
    scores["overall"] = _clip(np.mean(vals)) if vals else None

    # ---- most frequent sound problems ----
    conf, low = Counter(), Counter()
    for w in scored:
        for p in w["phones"]:
            if p["score"] < 45:
                low[p["p"]] += 1
                if p["heard"] != p["p"]:
                    conf[(p["p"], p["heard"])] += 1
    problems = []
    for ph, n in low.most_common(6):
        heard = [h for (e, h), _ in conf.most_common() if e == ph][:2]
        problems.append({"phone": ph, "count": n, "heard_as": heard, "tip": PHONE_TIPS.get(ph, ""),
                         "examples": [w["text"] for w in scored if any(p["p"] == ph and p["score"] < 45 for p in w["phones"])][:4]})

    return {
        "mode": "read" if read else "free",
        "register": register,
        "accent": accent,
        "target_text": target_text if read else "",
        "prompt": prompt,
        "transcript": asr["text"],
        "words": words,
        "fluency": fl,
        "intonation": into,
        "accuracy": accuracy,
        "reference": ref,
        "register_checks": checks,
        "sound_problems": problems,
        "scores": scores,
        "pitch": prosody.contour_for_plot(tr),
        "register_notes": reg["notes"],
    }
