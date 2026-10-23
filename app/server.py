"""Local web server: practice UI + analysis API.  Run with ./run.sh then open http://127.0.0.1:8765"""
import json
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import analysis, llm, tts
from .audio_asr import load16k, to_wav16k
from .locks import MODEL_LOCK
from .registers import REGISTERS
from .library import db as libdb, jobs as libjobs, sources as libsources

ROOT = Path(__file__).resolve().parent
SESSIONS = ROOT.parent / "data" / "sessions"
SESSIONS.mkdir(parents=True, exist_ok=True)

LOCK = MODEL_LOCK

app = FastAPI(title="Speak Coach")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/config")
def config():
    return {
        "registers": {k: {"label": v["label"], "notes": v["notes"], "wpm": v["wpm"], "style": v["style"]} for k, v in REGISTERS.items()},
        "llm": llm.LLM_MODEL,
    }


class TTSReq(BaseModel):
    text: str
    register: str = "work"
    accent: str = "us"
    gender: str = "female"
    voice: str | None = None


@app.post("/api/tts")
def make_tts(r: TTSReq):
    if not r.text.strip():
        raise HTTPException(400, "empty text")
    with LOCK:
        key, _ = tts.synth(r.text.strip(), r.register, r.accent, r.gender, r.voice)
    return {"id": key, "url": f"/api/tts/{key}.wav"}


@app.get("/api/tts/{key}.wav")
def get_tts(key: str):
    p = tts.path_for(key)
    if not p:
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/wav")


class GenReq(BaseModel):
    register: str = "work"
    topic: str = ""
    level: str = "B2"
    text: str = ""


@app.post("/api/sentences")
def sentences(r: GenReq):
    with LOCK:
        return {"items": llm.practice_sentences(r.register, r.topic, r.level)}


@app.post("/api/rewrite")
def rewrite(r: GenReq):
    with LOCK:
        return llm.rewrite_registers(r.text)


@app.post("/api/task")
def task(r: GenReq):
    with LOCK:
        return llm.speaking_task(r.register, r.level)


@app.post("/api/analyze")
def analyze(
    audio: UploadFile = File(...),
    mode: str = Form("read"),
    register: str = Form("work"),
    accent: str = Form("us"),
    target_text: str = Form(""),
    prompt: str = Form(""),
    ref_id: str = Form(""),
):
    if register not in REGISTERS:
        raise HTTPException(400, "unknown register")
    sid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    d = SESSIONS / sid
    d.mkdir()
    raw = d / f"input{Path(audio.filename or 'a.webm').suffix or '.webm'}"
    raw.write_bytes(audio.file.read())
    wav = to_wav16k(str(raw), str(d / "audio.wav"))
    if len(wav) < 16000 * 0.5:
        raise HTTPException(400, "Recording is too short.")
    ref_path, ref_url = None, None
    if ref_id.startswith("lib:"):
        clip = libdb.get(ref_id[4:])
        if clip:
            ref_path, ref_url = libdb.CLIPS_DIR / clip["path"], f"/api/library/clip/{clip['id']}/audio"
            libdb.bump(clip["id"], "practiced")
    elif ref_id:
        ref_path = tts.path_for(ref_id)
        ref_url = f"/api/tts/{ref_id}.wav" if ref_path else None
    ref = load16k(str(ref_path)) if ref_path else None
    with LOCK:
        result = analysis.analyze(wav, mode, register, accent, target_text, prompt, ref)
    result["id"] = sid
    result["audio_url"] = f"/api/sessions/{sid}/audio.wav"
    result["ref_url"] = ref_url
    result["ref_id"] = ref_id
    result["created"] = time.time()
    (d / "result.json").write_text(json.dumps(result, ensure_ascii=False))
    return result


@app.post("/api/feedback/{sid}")
def coach_feedback(sid: str):
    p = SESSIONS / Path(sid).name / "result.json"
    if not p.exists():
        raise HTTPException(404)
    result = json.loads(p.read_text())
    with LOCK:
        fb = llm.feedback(result)
    result["coach"] = fb
    p.write_text(json.dumps(result, ensure_ascii=False))
    return fb


@app.get("/api/sessions/{sid}/audio.wav")
def session_audio(sid: str):
    p = SESSIONS / Path(sid).name / "audio.wav"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/wav")


@app.get("/api/sessions/{sid}")
def session(sid: str):
    p = SESSIONS / Path(sid).name / "result.json"
    if not p.exists():
        raise HTTPException(404)
    return json.loads(p.read_text())


@app.get("/api/history")
def history():
    items = []
    for p in sorted(SESSIONS.glob("*/result.json"), reverse=True)[:200]:
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        items.append({
            "id": r["id"], "created": r.get("created"), "register": r["register"], "mode": r["mode"],
            "text": (r.get("target_text") or r.get("transcript") or "")[:90], "scores": r["scores"],
            "cefr": (r.get("coach") or {}).get("cefr"),
        })
    return items


@app.on_event("startup")
def warm():
    # Load the speech models in the background so the first analysis is quick.
    def _w():
        with LOCK:
            analysis.scorer()
        libjobs.resume()
    threading.Thread(target=_w, daemon=True).start()


# ---------- real-speech reference library ----------
@app.get("/api/library/search")
def library_search(q: str = "", register: str = "", source: str = "", tone: str = "", act: str = "", gender: str = "",
                   accent: str = "", length: str = "", pace: str = "", favorites: bool = False, order: str = "best",
                   limit: int = 30, offset: int = 0, any_words: bool = False):
    rows, total = libdb.search(q, register, source, tone, act, gender, accent, length, pace, favorites, order,
                               min(limit, 100), offset, any_words)
    return {"total": total, "items": rows}


@app.get("/api/library/stats")
def library_stats():
    return {**libdb.stats(), "catalog": libsources.CATALOG}


@app.get("/api/library/clip/{cid}/audio")
def library_audio(cid: str):
    c = libdb.get(cid)
    if not c:
        raise HTTPException(404)
    return FileResponse(libdb.CLIPS_DIR / c["path"], media_type="audio/flac")


@app.post("/api/library/clip/{cid}/favorite")
def library_fav(cid: str):
    libdb.bump(cid, "favorite")
    return libdb.get(cid)


class JobReq(BaseModel):
    kind: str
    source: str = ""
    max: int = 500
    src: str = ""
    title: str | None = None


@app.post("/api/library/jobs")
def library_job(r: JobReq):
    if r.kind == "ingest":
        if r.source not in libsources.CATALOG:
            raise HTTPException(400, "unknown source")
        return libjobs.submit("ingest", source=r.source, max=r.max)
    if r.kind == "import":
        if not r.src.strip():
            raise HTTPException(400, "give a file path or URL")
        return libjobs.submit("import", src=r.src.strip(), title=r.title or None)
    raise HTTPException(400, "unknown job kind")


@app.get("/api/library/jobs")
def library_jobs():
    return libjobs.all_jobs()


@app.post("/api/library/jobs/{jid}/stop")
def library_stop(jid: str):
    libjobs.stop(jid)
    return {"ok": True}
