#!/usr/bin/env python3
"""
hfu.py - Hugging Face URL downloader

Turns a Hugging Face URL / path into an `hf download` command and runs it,
downloading into the shared hub cache (exactly like calling `hf download`).

For GGUF repos it lists the available quants and lets you pick one (or
several); for vision models the matching mmproj file is added automatically.

Usage:
    hfu <url_or_path> [-q QUANT ...] [--mmproj MODE] [hf download options...]

Supported inputs:
    https://huggingface.co/<org>/<name>[/tree/...]
    https://huggingface.co/datasets/<org>/<name>
    https://huggingface.co/spaces/<org>/<name>
    <org>/<name>   datasets/<org>/<name>   models/<org>/<name>   spaces/<org>/<name>

Examples:
    hfu https://huggingface.co/zai-org/GLM-OCR
    hfu datasets/davanstrien/enc-brit-glm-ocr-v2-full
    hfu unsloth/Qwen3.6-35B-A3B-GGUF                  # interactive quant picker
    hfu unsloth/Qwen3.6-35B-A3B-GGUF -q UD-Q4_K_S     # no prompt
    hfu unsloth/Qwen3.6-35B-A3B-GGUF -q Q8_0 --mmproj BF16 --revision main
"""

import os
import re
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RepoFile:
    """A file in a Hub repo: relative path and size in bytes (None if unknown)."""
    path: str
    size: int | None = None


@dataclass
class Quant:
    """A selectable quantization: one or more .gguf files sharing a quant tag."""
    name: str
    files: list[str] = field(default_factory=list)
    size: int = 0


# --------------------------------------------------------------------------- #
# GGUF analysis (pure functions)
# --------------------------------------------------------------------------- #

_SHARD_SUFFIX = re.compile(r"-\d{5}-of-\d{5}$")


def is_gguf(path: str) -> bool:
    return path.lower().endswith(".gguf")


def is_mmproj(path: str) -> bool:
    return is_gguf(path) and "mmproj" in os.path.basename(path).lower()


def _stem(path: str) -> str:
    """Basename without .gguf extension and without a -0000N-of-0000M shard suffix."""
    base = os.path.basename(path)
    base = base[: -len(".gguf")]
    return _SHARD_SUFFIX.sub("", base)


def _common_model_prefix(stems: list[str]) -> str:
    """
    Longest prefix shared by all stems, trimmed back to the last '-' or '.'
    so we never split inside a quant token (e.g. 'Q4_K_' vs 'Q4_K_M').
    """
    prefix = os.path.commonprefix(stems)
    cut = max(prefix.rfind("-"), prefix.rfind("."))
    return prefix[: cut + 1] if cut >= 0 else ""


def group_quants(files: list[RepoFile]) -> list[Quant]:
    """
    Group the weight .gguf files of a repo into selectable quants.

    mmproj files and non-.gguf files are ignored. Sharded quants
    (…-00001-of-00002.gguf) collapse into a single Quant whose size is the
    sum of its shards. Result is sorted by size ascending.
    """
    weights = [f for f in files if is_gguf(f.path) and not is_mmproj(f.path)]
    if not weights:
        return []

    stems = [_stem(f.path) for f in weights]
    prefix = _common_model_prefix(stems) if len(stems) > 1 else ""

    groups: OrderedDict[str, Quant] = OrderedDict()
    for f, stem in zip(weights, stems):
        name = stem.removeprefix(prefix)
        name = name.lstrip("-._") or stem
        q = groups.setdefault(name, Quant(name=name))
        q.files.append(f.path)
        q.size += f.size or 0

    for q in groups.values():
        q.files.sort()
    return sorted(groups.values(), key=lambda q: (q.size, q.name))


_MMPROJ_PREFERENCE = ("F16", "BF16", "F32")


def _precision_tokens(path: str) -> set[str]:
    return {t.upper() for t in re.split(r"[-_.]", os.path.basename(path)) if t}


def pick_mmproj(mmproj_files: list[str], mode: str | None) -> list[str]:
    """
    Choose which mmproj file(s) to download.

    mode None  -> auto: prefer F16, then BF16, then F32, else first (sorted)
    mode 'all' -> every mmproj file
    mode 'none'-> nothing
    mode 'X'   -> the file(s) whose name contains precision token X (case-insensitive)
    """
    files = sorted(mmproj_files)
    if not files:
        return []
    if mode is None:
        for pref in _MMPROJ_PREFERENCE:
            hits = [f for f in files if pref in _precision_tokens(f)]
            if hits:
                return hits[:1]
        return files[:1]
    key = mode.strip().lower()
    if key == "all":
        return files
    if key == "none":
        return []
    hits = [f for f in files if key.upper() in _precision_tokens(f)]
    if not hits:
        raise ValueError(
            f"No mmproj file matching '{mode}'. Available: {', '.join(files)}"
        )
    return hits


def select_quants(quants: list[Quant], specs: list[str]) -> list[Quant]:
    """
    Resolve user-supplied quant names (from -q) against the grouped quants.

    A spec matches a quant if it equals the quant name (case-insensitive) or
    if every file stem of that quant ends with '-<spec>' / '.<spec>' / '_<spec>'
    (covers repos where prefix trimming removed a shared token such as 'UD-').
    """
    chosen: list[Quant] = []
    for spec in specs:
        key = spec.strip().lower()
        hit = next((q for q in quants if q.name.lower() == key), None)
        if hit is None:
            hit = next(
                (q for q in quants
                 if all(_stem(f).lower().endswith(("-" + key, "." + key, "_" + key)) for f in q.files)),
                None,
            )
        if hit is None:
            raise ValueError(
                f"No quant matching '{spec}'. Available: {', '.join(q.name for q in quants)}"
            )
        if hit not in chosen:
            chosen.append(hit)
    return chosen


def parse_choice(text: str, count: int) -> list[int] | None:
    """
    Parse the picker's input: '3' or '1,3 5' -> zero-based indices (deduped,
    in order). Returns None for quit ('q' or empty). Raises ValueError on
    garbage or out-of-range numbers.
    """
    text = text.strip()
    if text == "" or text.lower() in ("q", "quit"):
        return None
    indices: list[int] = []
    for tok in re.split(r"[,\s]+", text):
        if not tok.isdigit():
            raise ValueError(f"Not a number: '{tok}'")
        n = int(tok)
        if not 1 <= n <= count:
            raise ValueError(f"{n} is out of range 1-{count}")
        if n - 1 not in indices:
            indices.append(n - 1)
    return indices


def plan_filenames(files: list[RepoFile], quants: list[Quant], mmproj_mode: str | None) -> list[str]:
    """
    Final list of repo paths to download for the chosen quants:
    quant shards, then the selected mmproj file(s) (vision models only),
    then README.md if the repo has one.
    """
    names: list[str] = []
    for q in quants:
        names += [f for f in q.files if f not in names]
    mmproj_all = [f.path for f in files if is_mmproj(f.path)]
    names += pick_mmproj(mmproj_all, mmproj_mode)
    readme = next((f.path for f in files if f.path.lower() == "readme.md"), None)
    if readme:
        names.append(readme)
    return names


def format_size(n: int | None) -> str:
    if n is None:
        return "?"
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            break
        value /= 1024
    if unit == "B":
        return f"{int(value)} B"
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return f"{text} {unit}"


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def build_command(repo_id: str, repo_type: str, filenames: list[str], extra: list[str]) -> list[str]:
    cmd = ["hf", "download", repo_id, *filenames]
    if repo_type != "model":
        cmd += ["--repo-type", repo_type]
    return cmd + list(extra)


_OWN_OPTIONS_WITH_VALUE = {"-q", "--quant", "--mmproj"}
_OWN_FLAGS = {"-h", "--help"}


def _split_own_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """Separate hfu's own options (+ the first positional) from hf-download passthrough."""
    own: list[str] = []
    extra: list[str] = []
    have_target = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in _OWN_FLAGS:
            own.append(a)
        elif a in _OWN_OPTIONS_WITH_VALUE:
            own += argv[i:i + 2]
            i += 1
        elif a.split("=", 1)[0] in _OWN_OPTIONS_WITH_VALUE:
            own.append(a)
        elif not a.startswith("-") and not have_target:
            own.append(a)
            have_target = True
        else:
            extra.append(a)
        i += 1
    return own, extra


def parse_args(argv: list[str]):
    import argparse

    parser = argparse.ArgumentParser(
        prog="hfu",
        description="Download a Hugging Face repo into the hub cache via `hf download`. "
                    "GGUF repos get an interactive quant picker (mmproj auto-included for vision models).",
        epilog="Any other option (e.g. --revision, --local-dir, --dry-run) is passed straight to `hf download`.",
    )
    parser.add_argument("target", help="URL, org/name, or datasets|models|spaces/org/name")
    parser.add_argument("-q", "--quant", action="append", default=[], metavar="NAME",
                        help="quant to download (repeatable or comma-separated); skips the picker")
    parser.add_argument("--mmproj", default=None, metavar="MODE",
                        help="F16|BF16|F32|all|none (default: auto = F16 > BF16 > F32)")
    own, extra = _split_own_args(argv)
    ns = parser.parse_args(own)
    ns.quant = [s for chunk in ns.quant for s in chunk.split(",") if s.strip()]
    return ns, extra


_HF_HOSTS = ("huggingface.co", "hf.co")
_TYPE_PREFIXES = {"datasets": "dataset", "models": "model", "spaces": "space"}
_NAME_PART = re.compile(r"^[\w\-\.]+$")


def parse_hf_input(input_str: str) -> tuple[str, str]:
    """
    Parse a Hugging Face URL or path and return (repo_id, repo_type).

    Accepts full URLs (https://huggingface.co/…, https://hf.co/…, with or
    without /tree/…, /blob/…, query strings), and short forms
    'org/name', 'datasets/org/name', 'models/org/name', 'spaces/org/name'.
    repo_type is 'model', 'dataset', or 'space'.
    """
    from urllib.parse import urlparse

    text = input_str.strip()
    if "://" in text:
        url = urlparse(text)
        if url.netloc.lower().removeprefix("www.") not in _HF_HOSTS:
            raise ValueError(f"Not a Hugging Face URL: '{input_str}'")
        path = url.path
    else:
        path = text

    parts = [p for p in path.split("/") if p]
    repo_type = "model"
    if parts and parts[0] in _TYPE_PREFIXES:
        repo_type = _TYPE_PREFIXES[parts.pop(0)]

    if len(parts) < 2 or not all(_NAME_PART.match(p) for p in parts[:2]):
        raise ValueError(
            f"Invalid Hugging Face path format: '{input_str}'\n"
            f"Expected: <namespace>/<name> (e.g. unsloth/Qwen3.6-35B-A3B-GGUF)"
        )
    return f"{parts[0]}/{parts[1]}", repo_type


def table_rows(quants: list[Quant]) -> list[tuple[str, str, str, str]]:
    """Rows for the picker: (#, name, size, shard note)."""
    return [
        (str(i), q.name, format_size(q.size), f"{len(q.files)} files" if len(q.files) > 1 else "")
        for i, q in enumerate(quants, 1)
    ]


# --------------------------------------------------------------------------- #
# IO: Hub lookup, picker, execution
# --------------------------------------------------------------------------- #

def fetch_repo_files(repo_id: str, repo_type: str) -> list[RepoFile]:
    """List every file in the repo with its size (one Hub API call)."""
    from huggingface_hub import HfApi

    info = HfApi().repo_info(repo_id, repo_type=repo_type, files_metadata=True)
    return [RepoFile(s.rfilename, s.size) for s in (info.siblings or [])]


def render_table(quants: list[Quant]) -> None:
    rows = table_rows(quants)
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(show_edge=False, pad_edge=False)
        table.add_column("#", justify="right", style="bold")
        table.add_column("Quant")
        table.add_column("Size", justify="right")
        table.add_column("", style="dim")
        for row in rows:
            table.add_row(*row)
        Console().print(table)
    except ImportError:
        w = max(len(r[1]) for r in rows)
        for idx, name, size, note in rows:
            print(f"  {idx:>3}  {name:<{w}}  {size:>10}  {note}")


def prompt_for_quants(quants: list[Quant]) -> list[Quant]:
    """Interactive numbered picker. Returns [] if the user quits."""
    while True:
        try:
            answer = input(f"Pick a quant [1-{len(quants)}, comma-separated ok, q to quit]: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return []
        try:
            idx = parse_choice(answer, len(quants))
        except ValueError as e:
            print(f"   {e}")
            continue
        return [] if idx is None else [quants[i] for i in idx]


def resolve_gguf_download(repo_id: str, files: list[RepoFile], quant_specs: list[str],
                          mmproj_mode: str | None) -> list[str] | None:
    """
    Decide which files to download from a GGUF repo. Returns None if the
    user quit the picker. Raises ValueError for bad -q / --mmproj values.
    """
    quants = group_quants(files)
    mmproj_all = [f.path for f in files if is_mmproj(f.path)]

    if quant_specs:
        chosen = select_quants(quants, quant_specs)
    else:
        print(f"📦 {repo_id}: {len(quants)} quants available")
        if mmproj_all:
            picked = pick_mmproj(mmproj_all, mmproj_mode)
            print(f"👁  vision model — will include: {', '.join(picked) or 'no mmproj (--mmproj none)'}")
        print()
        render_table(quants)
        print()
        if not sys.stdin.isatty():
            raise ValueError("stdin is not a terminal; pass -q <quant> to choose non-interactively")
        chosen = prompt_for_quants(quants)
        if not chosen:
            return None

    return plan_filenames(files, chosen, mmproj_mode)


def main() -> None:
    ns, extra = parse_args(sys.argv[1:])

    try:
        repo_id, repo_type = parse_hf_input(ns.target)
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    filenames: list[str] = []
    if repo_type == "model":
        try:
            files = fetch_repo_files(repo_id, repo_type)
        except Exception as e:  # network / auth / 404 — fall back to plain hf download
            print(f"⚠️  Could not list repo files ({type(e).__name__}: {e}); downloading whole repo", file=sys.stderr)
            files = []
        if any(is_gguf(f.path) and not is_mmproj(f.path) for f in files):
            try:
                planned = resolve_gguf_download(repo_id, files, ns.quant, ns.mmproj)
            except ValueError as e:
                print(f"❌ {e}", file=sys.stderr)
                sys.exit(1)
            if planned is None:
                print("Aborted.")
                sys.exit(0)
            filenames = planned
        elif ns.quant or ns.mmproj:
            print("⚠️  -q/--mmproj ignored: repo has no .gguf files", file=sys.stderr)

    cmd = build_command(repo_id, repo_type, filenames, extra)

    print(f"📦 Downloading {repo_type}: {repo_id}")
    if filenames:
        total = sum((f.size or 0) for f in files if f.path in filenames)
        print(f"📄 {len(filenames)} files, {format_size(total)}")
    print(f"🔧 Command: {' '.join(cmd)}")
    print()

    try:
        subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print("\n✅ Successfully downloaded")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Download failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("❌ 'hf' CLI not found. Install with:", file=sys.stderr)
        print("   pip install 'huggingface_hub[cli]'  OR", file=sys.stderr)
        print("   curl -LsSf https://hf.co/cli/install.sh | bash", file=sys.stderr)
        sys.exit(127)


if __name__ == "__main__":
    main()
