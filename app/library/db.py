"""SQLite index of real-speech reference clips."""
import os
import sqlite3
import threading
from pathlib import Path

LIB_DIR = Path(os.environ.get("SPEAKCOACH_LIBRARY", Path(__file__).resolve().parents[2] / "data" / "library"))
CLIPS_DIR = LIB_DIR / "clips"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = LIB_DIR / "library.db"
_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,          -- meld, ami, earnings22, voxpopuli, expresso, edacc, media:<title>
  source_label TEXT,             -- e.g. "Friends S03E19", "AMI meeting ES2004a", a movie title
  speaker TEXT, gender TEXT, accent TEXT,
  text TEXT NOT NULL,            -- Whisper transcript (cased + punctuated)
  orig_text TEXT,                -- transcript shipped with the dataset / subtitle
  context TEXT,                  -- previous line(s) of the conversation, "Speaker: text"
  duration REAL, wpm INTEGER, pitch_sd REAL, pitch_range REAL, median_hz INTEGER, final_contour TEXT,
  emotion TEXT, style TEXT,      -- labels shipped with the dataset (MELD emotion, Expresso style)
  register TEXT, tone TEXT, act TEXT, usefulness INTEGER,   -- LLM labels
  quality INTEGER, pron_score INTEGER, wer REAL, snr_db REAL,
  path TEXT NOT NULL,
  favorite INTEGER DEFAULT 0, practiced INTEGER DEFAULT 0,
  created REAL DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS clips_reg ON clips(register);
CREATE INDEX IF NOT EXISTS clips_src ON clips(source);
CREATE VIRTUAL TABLE IF NOT EXISTS clips_fts USING fts5(text, context, content='clips', content_rowid='rowid');
CREATE TRIGGER IF NOT EXISTS clips_ai AFTER INSERT ON clips BEGIN
  INSERT INTO clips_fts(rowid, text, context) VALUES (new.rowid, new.text, new.context);
END;
CREATE TRIGGER IF NOT EXISTS clips_ad AFTER DELETE ON clips BEGIN
  INSERT INTO clips_fts(clips_fts, rowid, text, context) VALUES('delete', old.rowid, old.text, old.context);
END;
CREATE TABLE IF NOT EXISTS seen (id TEXT PRIMARY KEY, accepted INTEGER, reason TEXT);
"""


def conn() -> sqlite3.Connection:
    c = getattr(_local, "c", None)
    if c is None:
        c = sqlite3.connect(DB_PATH, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript(SCHEMA)
        _local.c = c
    return c


COLS = [
    "id", "source", "source_label", "speaker", "gender", "accent", "text", "orig_text", "context", "duration", "wpm",
    "pitch_sd", "pitch_range", "median_hz", "final_contour", "emotion", "style", "register", "tone", "act",
    "usefulness", "quality", "pron_score", "wer", "snr_db", "path",
]


def insert_clip(row: dict):
    c = conn()
    c.execute(
        f"INSERT OR REPLACE INTO clips ({','.join(COLS)}) VALUES ({','.join('?' * len(COLS))})",
        [row.get(k) for k in COLS],
    )
    c.execute("INSERT OR REPLACE INTO seen VALUES (?, 1, '')", (row["id"],))
    c.commit()


def mark_rejected(cid: str, reason: str):
    c = conn()
    c.execute("INSERT OR REPLACE INTO seen VALUES (?, 0, ?)", (cid, reason))
    c.commit()


def seen(cid: str) -> bool:
    return conn().execute("SELECT 1 FROM seen WHERE id=?", (cid,)).fetchone() is not None


def unlabeled(limit: int = 200):
    return [dict(r) for r in conn().execute(
        "SELECT id, source, source_label, speaker, text, context, emotion, style FROM clips WHERE register IS NULL LIMIT ?", (limit,))]


def set_labels(cid: str, register: str, tone: str, act: str, usefulness: int):
    c = conn()
    c.execute("UPDATE clips SET register=?, tone=?, act=?, usefulness=? WHERE id=?", (register, tone, act, usefulness, cid))
    c.commit()


def search(q: str = "", register: str = "", source: str = "", tone: str = "", act: str = "", gender: str = "",
           accent: str = "", length: str = "", pace: str = "", favorites: bool = False, order: str = "best",
           limit: int = 30, offset: int = 0, any_words: bool = False):
    where, args = ["register IS NOT NULL"], []
    if q.strip():
        where.append("rowid IN (SELECT rowid FROM clips_fts WHERE clips_fts MATCH ?)")
        args.append((" OR " if any_words else " ").join(f'"{t}"' for t in q.replace('"', " ").split()))
    for col, val in (("register", register), ("gender", gender), ("act", act)):
        if val:
            where.append(f"{col}=?")
            args.append(val)
    if source:
        where.append("(source=? OR source LIKE ?)")
        args += [source, source + ":%"]
    if tone:
        where.append("tone LIKE ?")
        args.append(f"%{tone}%")
    if accent:
        where.append("accent LIKE ?")
        args.append(f"%{accent}%")
    if length == "short":
        where.append("duration < 3.5")
    elif length == "medium":
        where.append("duration BETWEEN 3.5 AND 7")
    elif length == "long":
        where.append("duration > 7")
    if pace == "slow":
        where.append("wpm < 140")
    elif pace == "normal":
        where.append("wpm BETWEEN 140 AND 190")
    elif pace == "fast":
        where.append("wpm > 190")
    if favorites:
        where.append("favorite=1")
    orders = {
        "best": "usefulness DESC, quality DESC",
        "random": "RANDOM()",
        "new": "created DESC",
        "least_practiced": "practiced ASC, usefulness DESC",
    }
    sql = f"SELECT * FROM clips WHERE {' AND '.join(where)} ORDER BY {orders.get(order, orders['best'])} LIMIT ? OFFSET ?"
    c = conn()
    rows = [dict(r) for r in c.execute(sql, args + [limit, offset])]
    total = c.execute(f"SELECT COUNT(*) FROM clips WHERE {' AND '.join(where)}", args).fetchone()[0]
    return rows, total


def get(cid: str):
    r = conn().execute("SELECT * FROM clips WHERE id=?", (cid,)).fetchone()
    return dict(r) if r else None


def bump(cid: str, col: str):
    c = conn()
    if col == "practiced":
        c.execute("UPDATE clips SET practiced = practiced + 1 WHERE id=?", (cid,))
    elif col == "favorite":
        c.execute("UPDATE clips SET favorite = 1 - favorite WHERE id=?", (cid,))
    c.commit()


def stats():
    c = conn()
    q = lambda sql: [dict(r) for r in c.execute(sql)]
    return {
        "total": c.execute("SELECT COUNT(*) FROM clips").fetchone()[0],
        "labeled": c.execute("SELECT COUNT(*) FROM clips WHERE register IS NOT NULL").fetchone()[0],
        "hours": round((c.execute("SELECT SUM(duration) FROM clips").fetchone()[0] or 0) / 3600, 2),
        "by_register": q("SELECT register, COUNT(*) n FROM clips WHERE register IS NOT NULL GROUP BY register"),
        "by_source": q("SELECT CASE WHEN source LIKE 'media:%' THEN 'your media' ELSE source END AS source, COUNT(*) n FROM clips GROUP BY 1 ORDER BY n DESC"),
        "tones": q("SELECT tone, COUNT(*) n FROM clips WHERE tone IS NOT NULL GROUP BY tone ORDER BY n DESC LIMIT 30"),
        "acts": q("SELECT act, COUNT(*) n FROM clips WHERE act IS NOT NULL GROUP BY act ORDER BY n DESC LIMIT 30"),
        "accents": q("SELECT accent, COUNT(*) n FROM clips GROUP BY accent ORDER BY n DESC LIMIT 30"),
        "media": q("SELECT DISTINCT source FROM clips WHERE source LIKE 'media:%'"),
    }
