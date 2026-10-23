"""Grow the reference library from the command line (use this when the web app is not running).

  uv run python -m app.library ingest meld --max 2000
  uv run python -m app.library import ~/Movies/Before.Sunrise.1995.mkv --title "Before Sunrise"
  uv run python -m app.library import "https://www.youtube.com/watch?v=…"
  uv run python -m app.library label
  uv run python -m app.library stats
"""
import argparse
import json
import time

from . import db, jobs, process, sources


def main():
    ap = argparse.ArgumentParser(prog="python -m app.library")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ingest")
    a.add_argument("source", choices=list(sources.CATALOG))
    a.add_argument("--max", type=int, default=500)
    b = sub.add_parser("import")
    b.add_argument("src")
    b.add_argument("--title")
    sub.add_parser("label")
    sub.add_parser("stats")
    args = ap.parse_args()

    if args.cmd == "stats":
        print(json.dumps(db.stats(), indent=2))
        return
    if args.cmd == "label":
        print("labelled", process.label_pending(progress=lambda d: print("labelled", d, flush=True)))
        return
    job = jobs.submit("ingest", source=args.source, max=args.max) if args.cmd == "ingest" else \
        jobs.submit("import", src=args.src, title=args.title)
    last = ""
    while True:
        j = jobs.JOBS[job["id"]]
        if j["message"] != last:
            print(j["message"], flush=True)
            last = j["message"]
        if j["status"] in ("done", "error", "stopped"):
            print(json.dumps(jobs.public(j)["reasons"]))
            break
        time.sleep(1)


if __name__ == "__main__":
    main()
