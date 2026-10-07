"""sqlite metadata store. Two tables: images, audio."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from gemma_lens import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL UNIQUE,
    caption TEXT
);

CREATE TABLE IF NOT EXISTS audio (
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL UNIQUE,
    category TEXT,
    fold INTEGER
);
"""


@contextmanager
def connect(db_path: Path | None = None):
    db_path = db_path or config.DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def insert_image(conn: sqlite3.Connection, filename: str, caption: str | None) -> int:
    cur = conn.execute(
        "INSERT OR IGNORE INTO images (filename, caption) VALUES (?, ?)",
        (filename, caption),
    )
    if cur.lastrowid:
        return cur.lastrowid
    row = conn.execute("SELECT id FROM images WHERE filename = ?", (filename,)).fetchone()
    return row["id"]


def insert_audio(
    conn: sqlite3.Connection, filename: str, category: str, fold: int
) -> int:
    cur = conn.execute(
        "INSERT OR IGNORE INTO audio (filename, category, fold) VALUES (?, ?, ?)",
        (filename, category, fold),
    )
    if cur.lastrowid:
        return cur.lastrowid
    row = conn.execute("SELECT id FROM audio WHERE filename = ?", (filename,)).fetchone()
    return row["id"]


def get_image(conn: sqlite3.Connection, image_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    return dict(row) if row else None


def get_audio(conn: sqlite3.Connection, audio_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM audio WHERE id = ?", (audio_id,)).fetchone()
    return dict(row) if row else None


def all_images(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM images ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def all_audio(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM audio ORDER BY id").fetchall()
    return [dict(r) for r in rows]
