# HuggingFace Hub Utilities

Two command-line tools for working with [Hugging Face Hub](https://huggingface.co) models, datasets, and spaces.

| Tool | Purpose |
|------|---------|
| `hfu` | Download any Hub repo from a URL, path, or `org/name` shorthand, then refresh the views |
| `hf-xfer` | Translate between a `--local-dir` flat copy and the shared hub cache; maintain LM Studio and Ollama views |

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

## Consumer views — LM Studio and Ollama

The HF hub cache is the only place model bytes live. `hf-xfer sync` maintains two derived views of it:

| View | What it writes |
|------|----------------|
| `lmstudio` | `<root>/<org>/<name>/<file>.gguf` symlinks into the cache (mmproj beside the weights) |
| `ollama` | `<root>/blobs/sha256-<hash>` symlinks plus manifests under `manifests/hf.co/<org>/<name>/<tag>`, and alias manifests |

Configure roots in `~/.config/hfhub/config.toml` (or `$HFHUB_CONFIG`):

```toml
[views.lmstudio]
root = "~/.lmstudio/models"

[views.ollama]
root = "/path/to/huggingface/ollama"   # set OLLAMA_MODELS to the same path

[views.ollama.aliases]
"qwen3.6:27b" = "unsloth/Qwen3.6-27B-GGUF:Qwen3.6-27B-UD-Q4_K_XL.gguf"
```

```bash
hf-xfer sync                 # dry run: what would change
hf-xfer sync --execute       # write links and manifests
hf-xfer view status          # owned / tombstoned / foreign per view, unlinked cache blobs
hf-xfer view remove <org/name>:<file> --execute   # remove from views and remember it (tombstone)
hf-xfer view add    <org/name>:<file> --execute   # bring it back
hf-xfer view adopt  <foreign key> --repo-id <org/name> --move --execute   # move a foreign file into the cache
hf-xfer dupes                # foreign files that look like a cache entry
hf-xfer view remove --foreign <foreign key> --execute
```

Rules:

- Dry run is the default. Nothing is written without `--execute`.
- Safety brakes: if the cache directory itself is missing, sync reports it and does nothing; if a plan would remove every entry sync owns in a view, that view is refused unless you pass `--allow-mass-removal`. A view whose root does not exist is skipped with a notice, never created.
- Deleting a model inside LM Studio or with `ollama rm` sticks: sync records a tombstone and does not recreate it.
- Deleting only a secondary file (an alias manifest, a projector link, an mmproj link) is treated as damage and repaired on the next sync; deleting the primary file (the weight link in LM Studio, the `hf.co` manifest or model blob link in Ollama) is what records a tombstone.
- Sync never deletes or overwrites anything it did not create, with one exception: an Ollama blob that no manifest references any more is removed the same way `ollama rm` would remove it. Files that appear in a view without HF backing are listed as *foreign*.
- Sharded GGUFs are not grouped: each shard is its own entry. LM Studio gets a link per shard beside its siblings; Ollama skips shards with a warning, because it does not load split GGUFs.
- `view remove --foreign <key>` is the only command that deletes something sync did not create; for Ollama it removes the manifest and any blob no other manifest references, and it requires `--execute`.
- Ollama manifests are built from `https://huggingface.co/v2/<org>/<name>/manifests/<file>` so templates and parameters match what `ollama pull hf.co/...` would produce, but the model layer always points at your local file, so cached files that are behind the Hub still work. The registry response and config are cached under `<root>/.hfhub-registry/`, so later `--offline` runs need no network for entries already seen; if the network is actually down, the affected entries are skipped with a notice rather than failing the run. Repos that do not exist on the Hub get a minimal manifest and rely on the chat template embedded in the GGUF.
- `hfu` runs `sync --execute` after every successful download; pass `--no-sync` to skip.

Ollama needs read access to the cache and write access to the view root. The view roots must exist before the first sync - sync skips a view whose root is missing rather than creating it, so `mkdir -p` them first. With Ollama running as its own user, put the view root on the same drive as the cache, `chgrp ollama` it, `chmod 2775`, and set `OLLAMA_MODELS` in a systemd override together with `RequiresMountsFor=<mount point>`.

---

## Requirements

- Python 3.13+
- `huggingface-hub >= 1.20`
- `hf-xet`
- `hf` CLI (for `hfu` only)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
