"""Background jobs that grow the library: one worker, so models are never loaded twice.
Each clip takes the model lock separately, so practice requests can interleave with ingestion."""
import json
import queue
import threading
import time
import traceback
import uuid
from collections import Counter

from . import db, media, process, sources

PENDING = db.LIB_DIR / "pending_jobs.json"

JOBS: dict[str, dict] = {}
# Two lanes: your own imports don't wait behind a long dataset download (clips still take turns on the models).
_queues = {"ingest": queue.Queue(), "import": queue.Queue()}
_workers: dict[str, threading.Thread] = {}


def _run_candidates(job, gen, max_new):
    for c in gen:
        if job["stop"]:
            job["status"] = "stopped"
            break
        ok, why = process.process(c)
        if ok:
            job["accepted"] += 1
        elif why != "seen":
            job["rejected"] += 1
            job["reasons"][why] += 1
        job["message"] = f'{job["accepted"]} clips kept, {job["rejected"]} rejected'
        if max_new and job["accepted"] >= max_new:
            break
        if job["accepted"] % 10 == 0:
            _persist()
        # Label as we go so new clips show up in the library quickly.
        if job["accepted"] and job["accepted"] % 30 == 0:
            process.label_pending(max_batches=2)


def _execute(job):
    job["status"], job["started"] = "running", time.time()
    log = lambda m: job.__setitem__("message", m)
    try:
        if job["kind"] == "ingest":
            src, n = job["args"]["source"], int(job["args"].get("max", 500))
            log(f"Downloading {sources.CATALOG[src]['label']} (first run can take a while)…")
            _run_candidates(job, sources.candidates(src, n, seed=int(time.time()) % 1000), n)
        elif job["kind"] == "import":
            _run_candidates(job, media.candidates(job["args"]["src"], job["args"].get("title"), log), None)
        log(f'{job["accepted"]} clips kept. Labelling situation and tone…')
        process.label_pending(progress=lambda d: log(f'{job["accepted"]} clips kept · labelled {d}'))
        if job["status"] == "running":
            job["status"] = "done"
        log(f'Finished: {job["accepted"]} clips kept, {job["rejected"]} rejected')
    except Exception as e:
        job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
        job["message"] = job["error"]
        traceback.print_exc()
    job["finished"] = time.time()
    _persist()


def _persist():
    """Remember unfinished jobs so a server restart picks them up again."""
    live = []
    for j in sorted(JOBS.values(), key=lambda j: j["created"]):
        if j["status"] in ("queued", "running") and not j["stop"]:
            args = dict(j["args"])
            if j["kind"] == "ingest":
                args["max"] = max(0, int(args.get("max", 500)) - j["accepted"])
                if not args["max"]:
                    continue
            live.append({"kind": j["kind"], "args": args})
    try:
        PENDING.write_text(json.dumps(live))
    except OSError:
        pass


def resume():
    try:
        pending = json.loads(PENDING.read_text())
    except (OSError, ValueError):
        return
    for j in pending:
        submit(j["kind"], **j["args"])


def _loop(q):
    while True:
        job = q.get()
        if not job["stop"]:
            _execute(job)
        else:
            job["status"] = "stopped"


def submit(kind: str, **args) -> dict:
    w = _workers.get(kind)
    if w is None or not w.is_alive():
        _workers[kind] = threading.Thread(target=_loop, args=(_queues[kind],), daemon=True)
        _workers[kind].start()
    job = {"id": uuid.uuid4().hex[:8], "kind": kind, "args": args, "status": "queued", "accepted": 0, "rejected": 0,
           "reasons": Counter(), "message": "Queued", "stop": False, "created": time.time()}
    JOBS[job["id"]] = job
    _queues[kind].put(job)
    _persist()
    return public(job)


def stop(jid: str):
    if jid in JOBS:
        JOBS[jid]["stop"] = True
        _persist()


def public(job: dict) -> dict:
    return {k: (dict(v.most_common(5)) if k == "reasons" else v) for k, v in job.items() if k != "stop"}


def all_jobs() -> list[dict]:
    return [public(j) for j in sorted(JOBS.values(), key=lambda j: -j["created"])]
