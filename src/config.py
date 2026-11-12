"""Configuration loader and global random-seed helper."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load YAML configuration from disk."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_global_seed(seed: int) -> None:
    """Fix random seeds for Python stdlib and NumPy for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

