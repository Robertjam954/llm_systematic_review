"""
Shared I/O utilities — config loading, path helpers, run tracking.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv
from loguru import logger


def load_env(env_path: Path | None = None) -> None:
    """Load .env file. Falls back to project root .env."""
    path = env_path or Path(__file__).parents[3] / ".env"
    if path.exists():
        load_dotenv(path)
        logger.debug(f"Loaded env from {path}")
    else:
        logger.warning(f".env not found at {path}; using system environment variables")


def project_root() -> Path:
    return Path(__file__).parents[3]


def data_dir() -> Path:
    return project_root() / "data"


def experiments_dir() -> Path:
    return project_root() / "experiments"


def generate_run_id(method: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    uid = uuid.uuid4().hex[:6]
    return f"{method}_{ts}_{uid}"


def log_experiment(run_record: dict, output_dir: Path | None = None) -> Path:
    """Persist a run record as a JSON file in experiments/."""
    out_dir = output_dir or experiments_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_record.get("run_id", generate_run_id("unknown"))
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(run_record, fh, indent=2, default=str)
    logger.info(f"Run record saved to {out_path}")
    return out_path


def load_experiment(run_id: str, experiments_dir_: Path | None = None) -> dict:
    d = experiments_dir_ or experiments_dir()
    path = d / f"{run_id}.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_gold_standard(path: Path | None = None) -> set[str]:
    p = path or data_dir() / "splits" / "gold_standard_set.csv"
    df = pd.read_csv(p)
    return set(df["doc_id"].dropna().astype(str).tolist())


def read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
