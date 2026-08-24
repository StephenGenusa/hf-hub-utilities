# HuggingFace Hub Utilities

Two command-line tools for working with [Hugging Face Hub](https://huggingface.co) models, datasets, and spaces.

| Tool | Purpose |
|------|---------|
| `hfu` | Download any Hub repo from a URL, path, or `org/name` shorthand. Don't make me think |
| `hf-xfer` | Bidirectionally translate between a `--local-dir` flat copy and the shared hub cache |

---

## Installation

Requires Python 3.13+ and [`uv`](https://docs.astral.sh/uv/) (or standard pip).

```bash
git clone https://github.com/stephengenusa/huggingfacehub.git
cd huggingfacehub
uv pip install -e .
```

Both `hfu` and `hf-xfer` are registered as console scripts and will be available on your `PATH` after installation.

**Runtime dependency:** `hfu` shells out to the `hf` CLI. Install it with:

```bash
pip install "huggingface_hub[cli]"
# or
curl -LsSf https://hf.co/cli/install.sh | bash
```

---

## `hfu` — Hub Downloader

Accepts any of the formats the Hub uses and calls `hf download` for you. Files land in the shared **hub cache** (`~/.cache/huggingface/hub`), exactly as if you had run `hf download` yourself.

### Supported Input Formats

```
https://huggingface.co/<org>/<name>            (…/tree/main, …/blob/…, ?query are fine)
https://huggingface.co/datasets/<org>/<name>
https://huggingface.co/spaces/<org>/<name>
https://hf.co/<org>/<name>
<org>/<name>
models/<org>/<name>
datasets/<org>/<name>
spaces/<org>/<name>
```

### Usage

```
hfu <url_or_path> [-q QUANT ...] [--mmproj MODE] [hf download options...]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-q`, `--quant NAME` | interactive picker | Quant to download from a GGUF repo. Repeatable or comma-separated (`-q Q8_0,UD-Q4_K_S`). Exact, case-insensitive match. |
| `--mmproj MODE` | auto | Which mmproj to include for vision models: `F16`, `BF16`, `F32`, `all`, or `none`. Auto prefers F16 → BF16 → F32. |
| *anything else* | — | Passed straight to `hf download` (`--revision`, `--local-dir`, `--dry-run`, `--token`, …). |

### GGUF repos: quant picker

When the repo contains `.gguf` weight files, `hfu` lists the available quants with their sizes (sharded quants are shown as one entry) and prompts for a choice:

```
$ hfu unsloth/Qwen3.6-35B-A3B-GGUF
📦 unsloth/Qwen3.6-35B-A3B-GGUF: 24 quants available
👁  vision model — will include: mmproj-F16.gguf

 # ┃ Quant        ┃     Size ┃
 1 │ UD-IQ1_M     │  9.36 GB │
 …
13 │ UD-Q4_K_S    │ 19.46 GB │
 …
24 │ BF16         │ 64.61 GB │ 2 files
Pick a quant [1-24, comma-separated ok, q to quit]: 13
🔧 Command: hf download unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-Q4_K_S.gguf mmproj-F16.gguf README.md
```

The chosen files are passed to `hf download` **by exact filename** (never globs), so `Q4_K_S` can't accidentally pull in `Q4_K_XL`. `README.md` is included when present. If the repo ships `mmproj-*.gguf` files (vision model), one is added automatically.

Non-GGUF models, datasets, and spaces are downloaded in full, as before. If stdin is not a terminal, the picker is skipped and you must pass `-q`.

### Examples

```bash
# Full URL — model (default type)
hfu https://huggingface.co/zai-org/GLM-OCR

# Dataset / space
hfu https://huggingface.co/datasets/davanstrien/enc-brit-glm-ocr-v2-full
hfu spaces/black-forest-labs/FLUX.1-schnell

# GGUF repo — interactive picker
hfu unsloth/Qwen3.6-35B-A3B-GGUF

# GGUF repo — scripted, with a specific mmproj precision
hfu unsloth/Qwen3.6-35B-A3B-GGUF -q UD-Q4_K_S --mmproj BF16

# Two quants at once, and see what would happen first
hfu unsloth/Qwen3.6-35B-A3B-GGUF -q Q8_0,UD-Q4_K_S --dry-run

# Opt out of the cache for one download
hfu unsloth/Qwen3.6-35B-A3B-GGUF -q Q8_0 --local-dir ./models/qwen
```

Download progress streams live to the terminal.

---

## `hf-xfer` — Cache Translator

Hugging Face supports two on-disk layouts:

- **Hub cache** (`~/.cache/huggingface/hub`) — content-addressed blobs and snapshot symlinks, shared across all repos and revisions.
- **Local-dir copy** (`hf download --local-dir`) — flat, path-addressed files, portable but not deduplicated.

`hf-xfer` translates between these two layouts in both directions.

> **Dry-run is the default.** Pass `--execute` to write anything to disk.

### Subcommands

#### `import` — local-dir → cache

Seeds the shared cache from a flat local-dir copy. Useful when you received a model outside the Hub (USB drive, rsync, cloud storage) and want `hf download` to treat it as already cached.

```
hf-xfer import --local-dir <path> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--local-dir` | *(required)* | Path to the flat local-dir copy |
| `--repo-id` | inferred from path | `org/name`, e.g. `google/gemma-3-4b-it` |
| `--repo-type` | `model` | `model`, `dataset`, or `space` |
| `--revision` | `main` | Branch, tag, or full commit hash |
| `--cache-dir` | auto-resolved | Override `HF_HUB_CACHE` / `HF_HOME` |
| `--execute` | dry-run | Actually write blobs, symlinks, and refs |
| `--move` | copy | Move files instead of copying; erase emptied dirs |
| `--hardlink` | copy | Hardlink instead of copying (fastest, zero extra space) |
| `--offline` | off | Skip Hub query; use local `.metadata` files only |
| `--no-verify` | off | Skip post-import `snapshot_download` completeness check |

**Repo-id inference:** if `--repo-id` is omitted, the last two path components of `--local-dir` are used (e.g. `.../models/google/gemma-3-4b-it` → `google/gemma-3-4b-it`).

```bash
# Dry run first — see what would happen
hf-xfer import --local-dir ~/models/google/gemma-3-4b-it

# Move files into the cache (no extra disk space)
hf-xfer import --local-dir ~/models/google/gemma-3-4b-it --move --execute

# Offline — no Hub connection, uses local .metadata files
hf-xfer import --local-dir ~/models/google/gemma-3-4b-it --offline --execute
```

After import, `hf download google/gemma-3-4b-it` will find all files in the cache and skip re-downloading them.

#### `export` — cache → local-dir

Materializes a flat copy of a cached repo, writing `huggingface/download` metadata so subsequent `hf download --local-dir` calls are incremental.

```
hf-xfer export --local-dir <path> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--local-dir` | *(required)* | Destination directory for the flat copy |
| `--repo-id` | inferred from path | `org/name` |
| `--repo-type` | `model` | `model`, `dataset`, or `space` |
| `--revision` | `main` | Branch, tag, or full commit hash |
| `--cache-dir` | auto-resolved | Override `HF_HUB_CACHE` / `HF_HOME` |
| `--execute` | dry-run | Actually write files |
| `--move` | copy | Move blobs out of the cache; erase emptied dirs |
| `--hardlink` | copy | Hardlink instead of copying |

```bash
# Dry run — see what would be exported
hf-xfer export --local-dir ~/export/gemma --repo-id google/gemma-3-4b-it

# Copy to a local-dir
hf-xfer export --local-dir ~/export/gemma --repo-id google/gemma-3-4b-it --execute

# Hardlink for speed when source and destination are on the same filesystem
hf-xfer export --local-dir ~/export/gemma --repo-id google/gemma-3-4b-it --hardlink --execute
```

### Cache Resolution

`hf-xfer` resolves the cache directory using the same precedence as `huggingface_hub` itself:

```
--cache-dir flag  >  HF_HUB_CACHE  >  HUGGINGFACE_HUB_CACHE  >  HF_HOME/hub  >  ~/.cache/huggingface/hub
```

### File Transfer Modes

| Mode | Flag | Space used | Notes |
|------|------|-----------|-------|
| Copy | *(default)* | 2× | Safe; original untouched |
| Move | `--move` | 1× | Source removed after transfer |
| Hardlink | `--hardlink` | 1× | Fastest; requires same filesystem |

On export with `--move`, blobs shared by multiple snapshot revisions are automatically copied instead of moved to avoid breaking other revisions.

---

## Requirements

- Python 3.13+
- `huggingface-hub >= 1.20`
- `hf-xet`
- `hf` CLI (for `hfu` only)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
