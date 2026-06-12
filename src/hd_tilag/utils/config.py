from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config with optional shallow inheritance."""
    path = Path(path)
    data = _read_yaml(path)
    parent = data.pop("inherits", None)
    if parent:
        parent_path = path.parent / parent
        base = load_config(parent_path)
        return deep_merge(base, data)
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml

        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise TypeError(f"Top-level config must be a mapping: {path}")
        return loaded
    except ModuleNotFoundError:
        return _read_simple_yaml(path)


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    """A tiny YAML reader for this repository's simple config files.

    It supports nested mappings via two-space indentation and scalar lists in
    block style. PyYAML is still the recommended dependency for real use.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, root)]
    pending_key: tuple[int, dict[str, Any], str] | None = None

    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        text = raw.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()

        container = stack[-1][1]
        if text.startswith("- "):
            if not isinstance(container, list):
                if pending_key is None:
                    raise ValueError(f"List item without list parent in {path}: {raw}")
                _, parent, key = pending_key
                new_list: list[Any] = []
                parent[key] = new_list
                stack.append((indent - 2, new_list))
                container = new_list
            container.append(_parse_scalar(text[2:].strip()))
            continue

        if ":" not in text:
            raise ValueError(f"Unsupported YAML line in {path}: {raw}")

        key, value = text.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not isinstance(container, dict):
            raise ValueError(f"Mapping entry inside list is unsupported in {path}: {raw}")

        if value == "":
            new_map: dict[str, Any] = {}
            container[key] = new_map
            pending_key = (indent, container, key)
            stack.append((indent, new_map))
        else:
            container[key] = _parse_scalar(value)
            pending_key = None

    return root


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip('"').strip("'")
