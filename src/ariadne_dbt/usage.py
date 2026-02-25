"""Usage logging and statistics for Ariadne.

Every MCP tool call is recorded to the ``usage_log`` table in the local
SQLite index.  ``UsageLogger`` writes those rows and computes the stats
surfaced by ``ariadne usage``.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any


class UsageLogger:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # ── Write ──────────────────────────────────────────────────────────────────

    def log(
        self,
        tool_name: str,
        *,
        task_text: str | None = None,
        intent: str | None = None,
        focus_model: str | None = None,
        pivot_count: int | None = None,
        token_estimate: int | None = None,
        duration_ms: int | None = None,
    ) -> int:
        """Insert one usage row. Returns the new row id."""
        ts = datetime.now(timezone.utc).isoformat()
        cur = self.conn.execute(
            """
            INSERT INTO usage_log
                (ts, tool_name, task_text, intent, focus_model,
                 pivot_count, token_estimate, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, tool_name, task_text, intent, focus_model,
             pivot_count, token_estimate, duration_ms),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def rate(self, log_id: int, rating: int, notes: str | None = None) -> None:
        """Attach a 1–5 rating to an existing log row."""
        self.conn.execute(
            "UPDATE usage_log SET rating = ?, notes = ? WHERE id = ?",
            (max(1, min(5, rating)), notes, log_id),
        )
        self.conn.commit()

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_stats(self, days: int = 30) -> dict[str, Any]:
        """Return a stats dict covering the last ``days`` days."""
        since = _days_ago_iso(days)

        total = self.conn.execute(
            "SELECT COUNT(*) FROM usage_log WHERE ts >= ?", (since,)
        ).fetchone()[0]

        by_tool = dict(
            self.conn.execute(
                "SELECT tool_name, COUNT(*) FROM usage_log WHERE ts >= ? GROUP BY tool_name",
                (since,),
            ).fetchall()
        )

        by_intent = dict(
            self.conn.execute(
                """SELECT intent, COUNT(*) FROM usage_log
                   WHERE ts >= ? AND intent IS NOT NULL GROUP BY intent""",
                (since,),
            ).fetchall()
        )

        avg_tokens = self.conn.execute(
            "SELECT AVG(token_estimate) FROM usage_log WHERE ts >= ? AND token_estimate IS NOT NULL",
            (since,),
        ).fetchone()[0]

        avg_ms = self.conn.execute(
            "SELECT AVG(duration_ms) FROM usage_log WHERE ts >= ? AND duration_ms IS NOT NULL",
            (since,),
        ).fetchone()[0]

        avg_rating = self.conn.execute(
            "SELECT AVG(rating) FROM usage_log WHERE ts >= ? AND rating IS NOT NULL",
            (since,),
        ).fetchone()[0]

        top_models = self.conn.execute(
            """SELECT focus_model, COUNT(*) AS c FROM usage_log
               WHERE ts >= ? AND focus_model IS NOT NULL
               GROUP BY focus_model ORDER BY c DESC LIMIT 10""",
            (since,),
        ).fetchall()

        daily = self.conn.execute(
            """SELECT substr(ts, 1, 10) AS day, COUNT(*) AS c
               FROM usage_log WHERE ts >= ?
               GROUP BY day ORDER BY day""",
            (since,),
        ).fetchall()

        token_trend = self.conn.execute(
            """SELECT substr(ts, 1, 10) AS day, AVG(token_estimate) AS avg_tok
               FROM usage_log WHERE ts >= ? AND token_estimate IS NOT NULL
               GROUP BY day ORDER BY day""",
            (since,),
        ).fetchall()

        return {
            "period_days": days,
            "total_calls": total,
            "by_tool": by_tool,
            "by_intent": by_intent,
            "avg_token_estimate": round(avg_tokens) if avg_tokens else None,
            "avg_duration_ms": round(avg_ms) if avg_ms else None,
            "avg_rating": round(avg_rating, 2) if avg_rating else None,
            "top_models": [{"model": r[0], "calls": r[1]} for r in top_models],
            "daily_calls": [{"date": r[0], "calls": r[1]} for r in daily],
            "daily_avg_tokens": [{"date": r[0], "avg_tokens": round(r[1])} for r in token_trend],
        }

    def recent_queries(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the ``limit`` most recent log rows."""
        rows = self.conn.execute(
            """SELECT id, ts, tool_name, task_text, intent, focus_model,
                      pivot_count, token_estimate, duration_ms, rating
               FROM usage_log ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _days_ago_iso(days: int) -> str:
    from datetime import timedelta
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.isoformat()
