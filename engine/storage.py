"""Small SQLite persistence layer for user supplied telemetry and analyses."""
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any


def _db_path() -> str:
    return os.environ.get("ROOTSIGHT_DB", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rootsight.sqlite3"))


def connect():
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS telemetry (id INTEGER PRIMARY KEY, incident_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL)")
    conn.execute("CREATE INDEX IF NOT EXISTS telemetry_incident ON telemetry(incident_id, kind)")
    conn.execute("CREATE TABLE IF NOT EXISTS analyses (id INTEGER PRIMARY KEY, incident_id TEXT NOT NULL, result TEXT NOT NULL, created_at TEXT NOT NULL)")
    conn.commit()
    return conn


def save_dataset(incident_id: str, datasets: dict[str, list[dict[str, Any]]]) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with connect() as conn:
        for kind, rows in datasets.items():
            if not rows:
                continue
            # Replacing a dataset is idempotent; uploading the same file again must
            # not multiply identical records or inflate downstream evidence scores.
            conn.execute("DELETE FROM telemetry WHERE incident_id=? AND kind=?", (incident_id, kind))
            conn.execute("INSERT INTO telemetry(incident_id,kind,payload,created_at) VALUES(?,?,?,?)", (incident_id, kind, json.dumps(rows, default=str), now))
    return sum(len(rows) for rows in datasets.values())


def load_dataset(incident_id: str) -> dict[str, list[dict[str, Any]]]:
    out = {k: [] for k in ("metrics", "logs", "traces", "dependencies")}
    with connect() as conn:
        rows = conn.execute("SELECT kind,payload FROM telemetry WHERE incident_id=? ORDER BY id", (incident_id,)).fetchall()
    for row in rows:
        out.setdefault(row["kind"], []).extend(json.loads(row["payload"]))
    return out


def save_analysis(incident_id: str, result: dict[str, Any]) -> int:
    with connect() as conn:
        cur = conn.execute("INSERT INTO analyses(incident_id,result,created_at) VALUES(?,?,?)", (incident_id, json.dumps(result, default=str), datetime.now(timezone.utc).isoformat()))
        return int(cur.lastrowid)


def list_analyses(limit: int = 50) -> list[dict[str, Any]]:
    with connect() as conn:
        return [dict(r) | {"result": json.loads(r["result"])} for r in conn.execute("SELECT id,incident_id,result,created_at FROM analyses ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]


def delete_incident(incident_id: str) -> int:
    with connect() as conn:
        cur = conn.execute("DELETE FROM telemetry WHERE incident_id=?", (incident_id,))
        return cur.rowcount
