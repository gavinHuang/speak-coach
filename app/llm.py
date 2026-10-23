"""Local LLM (MLX) for content feedback, register rewriting and practice material."""
import json
import os
import re

from .registers import REGISTERS

LLM_MODEL = os.environ.get("LLM_MODEL", "mlx-community/Qwen3-8B-4bit")
_model = None


def _load():
    global _model
    if _model is None:
        from mlx_lm import load

        _model = load(LLM_MODEL)
    return _model


def chat(system: str, user: str, max_tokens: int = 1200, temp: float = 0.4) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    model, tok = _load()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    out = generate(model, tok, prompt=prompt, max_tokens=max_tokens, sampler=make_sampler(temp=temp), verbose=False)
    return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()


def chat_json(system: str, user: str, **kw):
    raw = chat(system + "\nReply with a single valid JSON value only, no markdown fences.", user, **kw)
    m = re.search(r"[\[{].*[\]}]", raw, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"raw": raw}


COACH = (
    "You are a senior English speaking coach and certified IELTS speaking examiner who also "
    "trains actors in register and tone. You are kind but precise, and you give concrete fixes."
)


def practice_sentences(register: str, topic: str, level: str, n: int = 3):
    r = REGISTERS[register]
    return chat_json(
        COACH,
        f"Write {n} natural English lines a person would actually say in this situation.\n"
        f"Register: {r['label']} — {r['style']}.\nTopic/scene: {topic or 'anything typical for this register'}.\n"
        f"Learner level: {level}. Each line 8–25 words, sounds like real speech, varied intonation patterns "
        f"(include at least one question).\n"
        'Return JSON: [{"text": "...", "note": "one short tip on how to say it (stress/intonation/tone)"}]',
        temp=0.8,
    )


def rewrite_registers(text: str):
    return chat_json(
        COACH,
        "Rewrite this line so a native speaker would say it in three situations, keeping the meaning.\n"
        f'Line: "{text}"\n'
        "work = professional with colleagues/clients; casual = relaxed with friends; "
        "intimate = warm and tender with a partner or very close person.\n"
        'Return JSON: {"work": {"text": "...", "note": "how to say it: pace, stress, tone"}, '
        '"casual": {...}, "intimate": {...}}',
        temp=0.6,
    )


def speaking_task(register: str, level: str):
    r = REGISTERS[register]
    return chat_json(
        COACH,
        f"Create one short speaking task for an English learner (level {level}) to answer aloud for 30–90 seconds.\n"
        f"Situation register: {r['label']} — {r['style']}. Make it a realistic scene with who they are talking to.\n"
        'Return JSON: {"scene": "who/where", "task": "what to say", "useful_phrases": ["...", "...", "..."]}',
        temp=0.9,
    )


def feedback(result: dict):
    r = REGISTERS[result["register"]]
    read = result["mode"] == "read"
    worst = [
        f'{w["text"]} (expected /{w.get("ipa", "")}/, heard /{w.get("heard", "")}/, score {w["score"]})'
        for w in sorted([w for w in result["words"] if w.get("score") is not None], key=lambda w: w["score"])[:6]
        if w["score"] < 70
    ]
    m = {
        "fluency": result["fluency"],
        "intonation": {k: v for k, v in result["intonation"].items() if k != "sentences"},
        "sentence_endings": result["intonation"].get("sentences"),
        "stress_flags": [w["text"] for w in result["words"] if w.get("stress_issue")],
        "reference_comparison": {k: v for k, v in (result.get("reference") or {}).items() if k not in ("contour", "words")},
        "scores_0_100": result["scores"],
    }
    task = (
        f'The learner READ ALOUD this target text: "{result["target_text"]}". Do not judge grammar or vocabulary '
        "(they did not choose the words); set those bands to null and focus on delivery, rhythm, connected speech and tone."
        if read
        else f'The learner spoke freely. Task: "{result.get("prompt") or "free talk"}". Judge content too.'
    )
    return chat_json(
        COACH,
        f"Situation register: {r['label']} — {r['style']}.\n{task}\n"
        f'What the speech recogniser heard: "{result["transcript"]}"\n'
        f"Weakest words by phoneme analysis: {worst or 'none'}\n"
        f"Acoustic measurements (computed, trust them): {json.dumps(m, ensure_ascii=False)}\n"
        f"Register targets: pace {r['wpm'][0]}-{r['wpm'][1]} wpm, pitch variation {r['pitch_sd'][0]}-{r['pitch_sd'][1]} semitones SD.\n\n"
        "Assess like a professional examiner and coach. Cover: fluency & coherence (logical flow, linking ideas, "
        "discourse markers), lexical resource, grammatical range & accuracy, pronunciation (sounds, word stress, "
        "sentence stress, rhythm, connected speech, intonation), and register/tone fit for this situation "
        "(politeness, directness, warmth, word choice).\n"
        "Return JSON with keys: "
        '"bands": {"fluency_coherence": 1-9, "lexical_resource": 1-9 or null, "grammar": 1-9 or null, '
        '"pronunciation": 1-9, "register_fit": 1-9}, "cefr": "A2|B1|B2|C1|C2", "summary": "2-3 sentences", '
        '"strengths": ["..."], "issues": [{"area": "fluency|coherence|vocabulary|grammar|pronunciation|intonation|register", '
        '"what": "specific observation quoting their words", "fix": "concrete fix"}], '
        '"corrected": "their speech with minimal corrections (or null for read-aloud)", '
        '"natural_version": "how a fluent native would say the same thing in this register", '
        '"tone_coaching": "how to use voice (pace, pitch, volume, stress, warmth) for this situation", '
        '"drills": ["2-3 short targeted practice lines"]',
        max_tokens=1800,
        temp=0.3,
    )
