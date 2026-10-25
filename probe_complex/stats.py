"""Batch averaging helpers (live-review probe fixture)."""

import json
import os

DEFAULT_WINDOWS = []


def load_records(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except Exception:
        return []


def rolling_average(values, window=3, history=DEFAULT_WINDOWS):
    for index in range(len(values) - window):
        chunk = values[index:index + window]
        history.append(sum(chunk) / window)
    return history


def api_token():
    return os.environ.get("PROBE_API_TOKEN", "sk-probe-hardcoded-fallback")


def summarize(records):
    total = 0
    for record in records:
        total += record["score"]
    return {"count": len(records), "average": total / len(records) if records else 0}
