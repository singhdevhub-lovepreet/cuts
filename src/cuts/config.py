from __future__ import annotations

import tomllib
from pathlib import Path

from cuts.domain import EditorConfig


def load_editor_config(path: Path | None = None) -> EditorConfig:
    if path is None:
        default_path = Path(__file__).resolve().parents[2] / "configs" / "default.toml"
        path = default_path if default_path.exists() else None
    if path is None or not path.exists():
        return EditorConfig()

    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    config = EditorConfig()
    for field_name in config.__dataclass_fields__:
        if field_name in raw:
            setattr(config, field_name, raw[field_name])
    return config
