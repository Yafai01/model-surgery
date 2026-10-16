"""Local SQLite store for scan results and quantization jobs."""

from __future__ import annotations

import gzip
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .schemas import ScanResult, ScanSummary

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id TEXT PRIMARY KEY,
    scanned_at TEXT NOT NULL,
    path TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    total_size INTEGER NOT NULL,
    n_params INTEGER,
    quant_label TEXT,
    valid INTEGER NOT NULL,
    result BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS scans_by_time ON scans(scanned_at DESC);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    data TEXT NOT NULL
);
"""


def _sql_int(v: object) -> int | None:
    """SQLite INTEGER is 64-bit; a crafted header can declare more elements than that. Store NULL instead of failing."""
    return v if isinstance(v, int) and not isinstance(v, bool) and -(2**63) <= v < 2**63 else None


class History:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)
            # Jobs that were running when the server stopped can never finish.
            for row in c.execute("SELECT id, data FROM jobs WHERE status IN ('queued','running','validating')").fetchall():
                data = json.loads(row[1])
                data["status"] = "interrupted"
                data["error"] = "server stopped while the job was running; output was not validated"
                c.execute("UPDATE jobs SET status=?, data=? WHERE id=?", ("interrupted", json.dumps(data), row[0]))

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=10)
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    # scans ---------------------------------------------------------------
    def save_scan(self, r: ScanResult) -> None:
        blob = gzip.compress(r.model_dump_json().encode("utf-8"), compresslevel=6)
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO scans VALUES (?,?,?,?,?,?,?,?,?,?)",
                (r.id, r.scanned_at, r.path, r.kind, r.name, _sql_int(r.total_size_bytes), _sql_int(r.fact_value("n_params")), r.quantization.label, int(r.integrity.valid), blob),
            )

    def get_scan(self, scan_id: str) -> ScanResult | None:
        with self._conn() as c:
            row = c.execute("SELECT result FROM scans WHERE id=?", (scan_id,)).fetchone()
        if row is None:
            return None
        return ScanResult.model_validate_json(gzip.decompress(row[0]))

    def list_scans(self, limit: int = 200) -> list[ScanSummary]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, scanned_at, path, kind, name, total_size, n_params, quant_label, valid FROM scans ORDER BY scanned_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            ScanSummary(id=r[0], scanned_at=r[1], path=r[2], kind=r[3], name=r[4], total_size_bytes=r[5], n_params=r[6], quant_label=r[7], valid=bool(r[8]))
            for r in rows
        ]

    def delete_scan(self, scan_id: str) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM scans WHERE id=?", (scan_id,)).rowcount > 0

    # jobs ----------------------------------------------------------------
    def save_job(self, job_id: str, created_at: str, status: str, data: dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO jobs VALUES (?,?,?,?)", (job_id, created_at, status, json.dumps(data)))

    def list_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT data FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
        return json.loads(row[0]) if row else None
