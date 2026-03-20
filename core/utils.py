from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


def resolve_repo_root(start: Path | None = None) -> Path:
    origin = (start or Path(__file__)).resolve()
    probe = origin if origin.is_dir() else origin.parent
    for parent in [probe, *probe.parents]:
        if (parent / "requirements.txt").is_file() and (parent / "AGENTS.md").is_file():
            return parent
    raise RuntimeError(f"Failed to locate repo root from {origin}")


def ensure_repo_root_on_path(start: Path | None = None) -> Path:
    repo_root = resolve_repo_root(start)
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.append(repo_root_str)
    return repo_root


def safe_token(value: object) -> str:
    token = str(value).strip().replace(".", "p")
    token = token.replace(" ", "_").replace("/", "_")
    return token


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

