from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

def ok(output_folder: Path | str, summary: str, key_files=None, metrics=None, **extra) -> dict:
    return {
        "status": "ok",
        "summary": summary,
        "output_folder": str(output_folder),
        "key_files": key_files or {},
        "metrics": metrics or {},
        **extra,
    }

def err(message: str) -> dict:
    return {"status": "error", "message": message}

def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return path

def write_run_summary(folder: Path, name: str, payload: dict[str, Any]) -> Path:
    payload = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **payload}
    return write_json(folder / f"{name}_run_summary.json", payload)

