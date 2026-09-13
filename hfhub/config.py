"""~/.config/hfhub/config.toml: view roots and Ollama aliases."""
from __future__ import annotations

import os
import re
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


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def save(config: Config) -> None:
    """Rewrite the whole file from the parsed model. Lossy: only for generated files."""
    _write(config.path, dump(config))


def _header_of(line: str) -> str:
    """The table header on this line, ignoring a trailing comment."""
    return re.sub(r"\s*#.*$", "", line).strip()


def _refuse_inline(text: str, view: str, path: Path) -> None:
    """Refuse when the aliases exist but not as a `[views.<view>.aliases]` table.

    An inline `aliases = {...}` (or any other spelling this line editor cannot
    see) would get a second, conflicting definition appended. A decode error here
    is left to _validate, which reports it properly.
    """
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return
    if data.get("views", {}).get(view, {}).get("aliases"):
        raise ValueError(f"config uses an inline aliases table; add the alias by hand: {path}")


def _validate(rendered: str, path: Path) -> None:
    """Never leave a config on disk that tomllib cannot read back."""
    try:
        tomllib.loads(rendered)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{path}: refusing to write unparseable TOML ({e})") from e


def add_alias(config: Config, view: str, alias: str, target: str) -> None:
    """Add or replace one alias line, leaving every other byte of the file alone.

    The config is hand-written: comments, key order, blank lines and keys this
    module does not model are the user's, so an alias added by `view adopt` is a
    text edit of the alias table, never a rewrite of the file from `dump`.
    """
    line = f"{_q(alias)} = {_q(target)}"
    header = f"[views.{view}.aliases]"
    text = config.path.read_text() if config.path.is_file() else ""
    lines = text.split("\n")
    start = next((i for i, l in enumerate(lines) if _header_of(l) == header), None)
    if start is None:
        _refuse_inline(text, view, config.path)
        body = text if not text or text.endswith("\n") else text + "\n"
        rendered = f"{body}\n{header}\n{line}\n"
    else:
        end = next((i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")), len(lines))
        # the same key may already be there bare, double- or single-quoted
        esc, raw = re.escape(_q(alias)[1:-1]), re.escape(alias)
        key = re.compile(rf"^\s*(?:\"(?:{esc}|{raw})\"|'{raw}'|{raw})\s*=")
        hit = next((i for i in range(start + 1, end) if key.match(lines[i])), None)
        if hit is not None:
            lines[hit] = line
        else:
            while end > start + 1 and not lines[end - 1].strip():
                end -= 1                      # insert before the blank line(s) that end the table
            lines.insert(end, line)
        rendered = "\n".join(lines)
    _validate(rendered, config.path)
    _write(config.path, rendered)
    config.views.setdefault(view, ViewConfig()).aliases[alias] = target
