"""JSONL read/write helpers with checkpoint-based resume support."""

from __future__ import annotations

import json
import os
from typing import Set


def load_existing_ids(path: str, id_field: str = "celex") -> Set[str]:
    """Return the set of *id_field* values already present in a JSONL file.

    If *path* does not exist or is empty, returns an empty set.
    """
    ids: Set[str] = set()
    if not os.path.exists(path):
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                val = record.get(id_field)
                if val is not None:
                    ids.add(str(val))
            except json.JSONDecodeError:
                continue
    return ids


def append_jsonl(path: str, record: dict) -> None:
    """Append a single JSON record to a JSONL file (newline-delimited)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    """Read all records from a JSONL file."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records
