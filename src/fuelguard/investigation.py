"""Parse and run the investigation queries in sql/investigation.sql over a feed.

Kept as a small module so both the command-line script and the tests use the same code:
the queries are named with `-- name:` markers and each carries a `--` description.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

SQL_FILE = Path(__file__).resolve().parents[2] / "sql" / "investigation.sql"


def parse_queries(sql_text: str) -> list[tuple[str, str, str]]:
    """Split the file into (name, description, sql) blocks on the `-- name:` markers."""
    parts = re.split(r"(?m)^--\s*name:\s*(\w+)\s*$", sql_text)
    blocks = []
    body_iter = iter(parts[1:])
    for name, body in zip(body_iter, body_iter, strict=False):
        lines = body.strip().split("\n")
        desc, i = [], 0
        while i < len(lines) and lines[i].strip().startswith("--"):
            desc.append(lines[i].strip("- ").strip())
            i += 1
        sql = "\n".join(lines[i:]).strip().rstrip(";")
        blocks.append((name, " ".join(desc), sql))
    return blocks


def run(csv_path: str | Path, sql_text: str | None = None,
        con: duckdb.DuckDBPyConnection | None = None) -> dict[str, tuple[str, pd.DataFrame]]:
    """Run every query against the CSV feed; return {name: (description, result)}."""
    if sql_text is None:
        sql_text = SQL_FILE.read_text()
    con = con or duckdb.connect()
    con.execute(f"CREATE OR REPLACE VIEW t AS SELECT * FROM read_csv_auto('{csv_path}')")
    return {name: (desc, con.execute(sql).df())
            for name, desc, sql in parse_queries(sql_text)}
