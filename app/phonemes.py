"""Phoneme-level pronunciation scoring.

Expected phonemes come from espeak-ng (via phonemizer). A wav2vec2 model fine-tuned
to recognise espeak IPA phones gives per-frame phone posteriors. We force-align the
expected phones to the audio with CTC Viterbi and score each phone by how strongly the
model hears it (a simplified Goodness-of-Pronunciation), and report what it heard instead.
"""
import json
import os


import numpy as np
import torch
from huggingface_hub import hf_hub_download
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

PHONE_MODEL = os.environ.get("PHONEME_MODEL", "facebook/wav2vec2-xlsr-53-espeak-cv-ft")
# The multilingual model is less biased toward dictionary English pronunciations, so it
# reports L2 errors (v→w, ɪ→iː, s→z) more faithfully than the English-only lv-60 model.
FRAME_SEC = 0.02
VOWEL_CHARS = set("aeiouɑɐɒæɔəɘɚɛɜɝɞɤɨɪʊʌʏøœɯɵᵻ")
STRESS_MARKS = {"ˈ": 1, "ˌ": 2}

# Phones that are acceptable variants of each other (accent / reduction differences).
EQUIV_GROUPS = [
    {"ə", "ɐ", "ʌ", "ᵻ"},
    {"ɪ", "ᵻ", "ə"},
    {"ɚ", "ɝ", "ɜː", "ɜ", "ə"},
    {"i", "iː"},
    {"u", "uː"},
    {"ɾ", "t", "d"},
    {"oʊ", "o", "əʊ", "oː"},
    {"ɑː", "ɑ", "ɒ", "ɔ", "ɔː", "a"},
    {"ɛ", "e"},
    {"eɪ", "e"},
    {"ɹ", "r"},
    {"l", "ɫ"},
    {"n̩", "n"},
    {"ʔ", "t"},
]

# Function words are normally reduced in connected speech ("to" → tə, "can" → kən).
WEAK_FORM_WORDS = set(
    "a an the to for of from at and but or as than that can could would should will shall was were "
    "is are am be been have has had do does you your he him his she her we us our they them their me my "
    "some there with just".split()
)
WEAK_VOWELS = ["ə", "ɐ", "ɪ", "ᵻ", "ʊ"]


def is_vowel(p: str) -> bool:
    return bool(p) and p[0] in VOWEL_CHARS


def ctc_force_align(logp: np.ndarray, tokens: list[int], blank: int):
    """Viterbi CTC alignment. Returns list of (first_frame, last_frame) per token, or None."""
    T, L = logp.shape[0], len(tokens)
    if L == 0 or T < L:
        return None
    S = 2 * L + 1
    ext = np.full(S, blank, dtype=np.int64)
    ext[1::2] = tokens
    allow_skip = np.zeros(S, dtype=bool)
    for s in range(2, S):
        allow_skip[s] = ext[s] != blank and ext[s] != ext[s - 2]
    NEG = -1e30
    dp = np.full(S, NEG)
    dp[0] = logp[0, blank]
    dp[1] = logp[0, ext[1]]
    back = np.zeros((T, S), dtype=np.int8)
    emit = logp[:, ext]  # T x S
    for t in range(1, T):
        stay = dp
        step = np.concatenate(([NEG], dp[:-1]))
        skip = np.where(allow_skip, np.concatenate(([NEG, NEG], dp[:-2])), NEG)
        cand = np.stack([stay, step, skip])
        idx = cand.argmax(0)
        dp = cand[idx, np.arange(S)] + emit[t]
        back[t] = idx
    s = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
    if dp[s] <= NEG / 2:
        return None
    states = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        states[t] = s
        s -= int(back[t, s])
    spans = [[None, None] for _ in range(L)]
    for t, st in enumerate(states):
        if st % 2 == 1:
            k = (st - 1) // 2
            if spans[k][0] is None:
                spans[k][0] = t
            spans[k][1] = t
    if any(a is None for a, _ in spans):
        return None
    return [tuple(x) for x in spans]


class PhonemeScorer:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(PHONE_MODEL)
        self.model = Wav2Vec2ForCTC.from_pretrained(PHONE_MODEL).to(self.device).eval()
        with open(hf_hub_download(PHONE_MODEL, "vocab.json")) as f:
            self.tok2id: dict[str, int] = json.load(f)
        self.id2tok = {v: k for k, v in self.tok2id.items()}
        self.blank = self.tok2id.get("<pad>", 0)
        self.special = [i for t, i in self.tok2id.items() if t.startswith("<") or t in ("|", " ")]
        self.equiv_ids: dict[int, list[int]] = {}
        for tok, tid in self.tok2id.items():
            ids = set()
            for g in EQUIV_GROUPS:
                if tok in g:
                    ids |= {self.tok2id[x] for x in g if x in self.tok2id and x != tok}
            if is_vowel(tok):  # r-coloured variants: model may emit "ɑːɹ" where espeak splits "ɑː ɹ"
                for x, xid in self.tok2id.items():
                    if x in (tok + "ɹ", tok.rstrip("ɹ")) and x != tok:
                        ids.add(xid)
            self.equiv_ids[tid] = sorted(ids)
        self.weak_ids = [self.tok2id[v] for v in WEAK_VOWELS if v in self.tok2id]
        self._backends: dict[str, EspeakBackend] = {}

    # ---------- expected phonemes ----------
    def _backend(self, lang: str) -> EspeakBackend:
        if lang not in self._backends:
            self._backends[lang] = EspeakBackend(
                lang, preserve_punctuation=False, with_stress=True, language_switch="remove-flags"
            )
        return self._backends[lang]

    def _to_vocab(self, p: str) -> list[str]:
        if p in self.tok2id:
            return [p]
        if p.replace("ː", "") in self.tok2id:
            return [p.replace("ː", "")]
        for g in EQUIV_GROUPS:
            if p in g:
                for alt in g:
                    if alt in self.tok2id:
                        return [alt]
        return [c for c in p if c in self.tok2id]

    def _raw(self, words: list[str], lang: str) -> list[str]:
        """Space-separated espeak phones per word, phonemized in context so that homographs
        ("she lives" vs "their lives") and weak forms ("a" → ɐ) come out right."""
        sep = Separator(phone=" ", word=" | ", syllable="")
        be = self._backend(lang)
        out = []
        for i in range(0, len(words), 25):
            chunk = words[i : i + 25]
            res = be.phonemize([" ".join(chunk)], separator=sep, strip=True, njobs=1)[0]
            parts = [x.strip() for x in res.split("|")]
            if len(parts) != len(chunk):  # numbers/abbreviations expand to several words: go word by word
                parts = [x.replace("|", " ").strip() for x in be.phonemize(chunk, separator=sep, strip=True, njobs=1)]
            out += parts
        return out

    def expected(self, words: list[str], lang: str) -> list[list[tuple[str, int]]]:
        """Per word: list of (phone, stress) where stress is 1 primary, 2 secondary, 0 none."""
        if not words:
            return []
        result = []
        for out in self._raw(words, lang):
            phones, pending = [], 0
            for raw in out.split():
                stress = 0
                for m, v in STRESS_MARKS.items():
                    if m in raw:
                        stress = v if not stress else min(stress, v)
                        raw = raw.replace(m, "")
                stress = stress or pending
                pending = 0
                for q in self._to_vocab(raw):
                    if is_vowel(q):
                        phones.append((q, stress))
                        stress = 0
                    else:
                        phones.append((q, 0))
                pending = stress  # stress mark landed on a consonant: carry it to the next vowel
            result.append(phones)
        return result

    def ipa(self, words: list[str], lang: str) -> list[str]:
        return [x.replace(" ", "") for x in self._raw(words, lang)] if words else []

    # ---------- acoustic model ----------
    def logprobs(self, audio: np.ndarray) -> np.ndarray:
        chunk = 16000 * 30
        outs = []
        for i in range(0, max(len(audio), 1), chunk):
            seg = audio[i : i + chunk]
            if len(seg) < 400:
                continue
            x = self.fe(seg, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
            with torch.no_grad():
                logits = self.model(x).logits[0].float()
            outs.append(torch.log_softmax(logits, -1).cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, len(self.tok2id)))

    # ---------- scoring ----------
    def score(self, audio: np.ndarray, words: list[str], lang: str = "en-us"):
        exp = self.expected(words, lang)
        logp = self.logprobs(audio)
        flat, owner = [], []
        for wi, phones in enumerate(exp):
            for pi, (p, _) in enumerate(phones):
                flat.append(self.tok2id[p])
                owner.append((wi, pi))
        spans = ctc_force_align(logp, flat, self.blank)
        out = [{"word": w, "phones": [], "start": None, "end": None, "score": None, "heard": ""} for w in words]
        if spans is None:
            return out, logp

        probs = np.exp(logp)
        nonblank = 1.0 - probs[:, self.blank : self.blank + 1] + 1e-6
        pn = probs / nonblank
        pn[:, self.blank] = 0
        pn[:, self.special] = 0
        T = logp.shape[0]

        for k, (tid, (wi, pi)) in enumerate(zip(flat, owner)):
            s, e = spans[k]
            region_end = spans[k + 1][0] if k + 1 < len(spans) else min(T, e + 6)
            seg = pn[s : e + 1]
            p_target = float(seg[:, tid].max())
            ok_ids = {tid, *self.equiv_ids.get(tid, [])}
            eq = self.equiv_ids.get(tid, [])
            if eq:
                p_target = max(p_target, 0.85 * float(seg[:, eq].max()))
            if is_vowel(self.id2tok[tid]) and words[wi] in WEAK_FORM_WORDS:
                p_target = max(p_target, 0.95 * float(seg[:, self.weak_ids].max()))
                ok_ids |= set(self.weak_ids)
            heard_id = int(seg.sum(0).argmax())
            p_target = min(1.0, p_target)
            # If the model's best guess is the right sound (or an accepted variant) the phone is
            # correct; posterior only grades how clear it was. Otherwise it is capped below 70.
            score = 70 + 30 * p_target if heard_id in ok_ids else 70 * p_target
            if words[wi] in WEAK_FORM_WORDS and not is_vowel(self.id2tok[tid]):
                score = max(score, 72)  # "and" → ən, "of" → ə, "with" → wɪθ/wɪð are all normal
            phone, stress = exp[wi][pi]
            out[wi]["phones"].append(
                {
                    "p": phone,
                    "stress": stress,
                    "score": round(score),
                    "heard": self.id2tok.get(heard_id, "?"),
                    "t0": round(s * FRAME_SEC, 3),
                    "t1": round(region_end * FRAME_SEC, 3),
                }
            )

        for w in out:
            ph = w["phones"]
            if not ph:
                continue
            w["start"], w["end"] = ph[0]["t0"], ph[-1]["t1"]
            scores = np.array([p["score"] for p in ph], dtype=float)
            # Weight the worst phone more: one badly wrong sound makes the whole word sound off.
            w["score"] = round(0.7 * scores.mean() + 0.3 * scores.min())
            f0, f1 = int(w["start"] / FRAME_SEC), int(w["end"] / FRAME_SEC)
            ids = logp[f0:f1].argmax(1)
            heard, prev = [], None
            for i in ids:
                if i != prev and i != self.blank and i not in self.special:
                    heard.append(self.id2tok[int(i)])
                prev = i
            w["heard"] = "".join(heard)
        return out, logp
