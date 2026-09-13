"""~/.config/hfhub/config.toml: view roots and Ollama aliases."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

VIEW_NAMES = ("lmstudio", "ollama")


@dataclass
class ViewConfig:
    root: Path | None = None
    aliases: dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    path: Path
    views: dict[str, ViewConfig]


def default_path() -> Path:
    env = os.environ.get("HFHUB_CONFIG")
    if env:
        return Path(env).expanduser()
    return Path("~/.config/hfhub/config.toml").expanduser()


def load(path: Path | None = None) -> Config:
    path = path or default_path()
    views = {name: ViewConfig() for name in VIEW_NAMES}
    if path.is_file():
        data = tomllib.loads(path.read_text())
        for name in VIEW_NAMES:
            section = data.get("views", {}).get(name, {})
            root = section.get("root")
            views[name] = ViewConfig(
                root=Path(root).expanduser() if root else None,
                aliases=dict(section.get("aliases", {})),
            )
    return Config(path=path, views=views)


def _q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def dump(config: Config) -> str:
    lines: list[str] = []
    for name in VIEW_NAMES:
        v = config.views[name]
        if v.root is None and not v.aliases:
            continue
        lines.append(f"[views.{name}]")
        if v.root is not None:
            lines.append(f"root = {_q(str(v.root))}")
        lines.append("")
        if v.aliases:
            lines.append(f"[views.{name}.aliases]")
            for alias, target in sorted(v.aliases.items()):
                lines.append(f"{_q(alias)} = {_q(target)}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save(config: Config) -> None:
    config.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.path.with_suffix(".tmp")
    tmp.write_text(dump(config))
    os.replace(tmp, config.path)
