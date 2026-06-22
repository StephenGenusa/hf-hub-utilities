# HuggingFace Hub Utilities

Two command-line tools for working with [Hugging Face Hub](https://huggingface.co) models, datasets, and spaces.

| Tool | Purpose |
|------|---------|
| `hfu` | Download any Hub repo from a URL, path, or `org/name` shorthand |
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

Accepts any of the formats the Hub uses and calls `hf download` for you.

### Supported Input Formats

```
https://huggingface.co/<org>/<name>
https://huggingface.co/datasets/<org>/<name>
https://huggingface.co/spaces/<org>/<name>
<org>/<name>
models/<org>/<name>
datasets/<org>/<name>
spaces/<org>/<name>
```

### Usage

```
hfu <url_or_path>
```

### Examples

```bash
# Full URL — model (default type)
hfu https://huggingface.co/zai-org/GLM-OCR

# Full URL — dataset
hfu https://huggingface.co/datasets/davanstrien/enc-brit-glm-ocr-v2-full

# Short path — model
hfu zai-org/GLM-OCR

# Short path — dataset
hfu datasets/davanstrien/enc-brit-glm-ocr-v2-full

# Short path — space
hfu spaces/black-forest-labs/FLUX.1-schnell
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
