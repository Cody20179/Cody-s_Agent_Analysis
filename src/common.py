from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def _iter_key_files(key_files) -> list[tuple[str, str]]:
    if not isinstance(key_files, dict):
        return []
    items: list[tuple[str, str]] = []
    for label, value in key_files.items():
        if isinstance(value, dict):
            items.extend((f"{label}.{child_label}", child_value) for child_label, child_value in _iter_key_files(value))
        elif isinstance(value, (str, Path)):
            items.append((str(label), str(value)))
    return items


def artifacts_from_key_files(key_files) -> list[dict[str, str]]:
    artifacts = []
    for label, value in _iter_key_files(key_files):
        path = Path(value)
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            artifacts.append({
                "type": "image",
                "label": label,
                "path": str(path),
            })
    return artifacts


def ok(output_folder: Path | str, summary: str, key_files=None, metrics=None, **extra) -> dict:
    artifacts = extra.pop("artifacts", None)
    if artifacts is None:
        artifacts = artifacts_from_key_files(key_files)
    return {
        "status": "ok",
        "summary": summary,
        "output_folder": str(output_folder),
        "key_files": key_files or {},
        "metrics": metrics or {},
        "artifacts": artifacts,
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
