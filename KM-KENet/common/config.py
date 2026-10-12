"""Configuration loading shared by all pipeline steps."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class PipelineConfig:
    """Read-only view over the YAML configuration."""

    raw: Mapping[str, Any]
    root: Path = PROJECT_ROOT

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if value is None:
            raise KeyError(f"Missing configuration section '{name}'")
        if not isinstance(value, dict):
            raise TypeError(f"Configuration section '{name}' must be a mapping")
        return value

    def path(self, key: str) -> Path:
        """Return an absolute path declared under the `paths` section."""
        paths = self.section("paths")
        if key not in paths:
            raise KeyError(f"Missing path '{key}' in the 'paths' section")
        return self.resolve(paths[key])

    def resolve(self, value: str | os.PathLike[str]) -> Path:
        path = Path(os.path.expandvars(os.path.expanduser(str(value))))
        return path if path.is_absolute() else (self.root / path).resolve()

    @property
    def seed(self) -> int:
        return int(self.raw.get("project", {}).get("seed", 42))

    @property
    def device(self) -> str:
        return str(self.raw.get("project", {}).get("device", "auto"))


def _set_nested(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = target
    for key in keys[:-1]:
        child = node.get(key)
        if child is None:
            child = node[key] = {}
        elif not isinstance(child, dict):
            raise ValueError(f"Cannot override '{dotted_key}': '{key}' is not a mapping")
        node = child
    node[keys[-1]] = value


def apply_overrides(raw: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply `section.key=value` overrides; values are parsed as YAML scalars."""
    result = copy.deepcopy(raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{item}', empty key")
        _set_nested(result, key, yaml.safe_load(value))
    return result


def load_config(
    path: str | os.PathLike[str] | None = None,
    overrides: Iterable[str] | None = None,
) -> PipelineConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute() and not config_path.exists():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")

    raw = apply_overrides(raw, overrides or [])
    return PipelineConfig(raw=raw, root=PROJECT_ROOT)
