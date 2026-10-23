from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class Config:
    def __init__(self, data: dict[str, Any], source: str | Path):
        self.data = data
        self.source = Path(source).resolve()
        self.base_dir = self.source.parent.parent if self.source.parent.name == "configs" else self.source.parent

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls(data, p)

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for key in dotted.split("."):
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    def copy(self) -> "Config":
        return Config(deepcopy(self.data), self.source)

    @property
    def dataset_root(self) -> Path:
        return Path(self.get("project.dataset_root"))

    @property
    def output_root(self) -> Path:
        p = Path(self.get("project.output_root", "outputs"))
        if p.is_absolute():
            return p
        return (self.base_dir / p).resolve()

    def output_path(self, *parts: str) -> Path:
        return self.output_root.joinpath(*parts)


def load_questions(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
