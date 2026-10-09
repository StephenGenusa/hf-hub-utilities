# hf-hub-utilities

Keep one copy of every model on disk, in the Hugging Face hub cache, and let
LM Studio and Ollama use it directly.

| Tool | Purpose |
|------|---------|
| `hfu` | Download any Hub repo from a URL, path, or `org/name`, with a quant picker for GGUF repos, then refresh the views |
| `hf-xfer` | Translate between a `--local-dir` copy and the hub cache; maintain the LM Studio and Ollama views; maintain the cache itself |

Every `hf-xfer` command that writes is a dry run unless you pass `--execute`.
`hfu` downloads straight away, like `hf download`; `--dry-run --no-sync`
previews the download without writing anything.

---

## Why

The Hugging Face cache, LM Studio, and Ollama all want the same GGUF files, and
each will happily keep its own copy. Ollama's blob store names files by their
SHA-256, and so does the HF cache, so an Ollama model can be a symlink into the
cache; LM Studio just needs an `org/name/file.gguf` tree, which can be symlinks
too. `hf-xfer sync` maintains both trees from the cache, and nothing else on the
machine needs to know.

```
$HF_HOME/hub/models--unsloth--Qwen3.6-27B-GGUF/blobs/<sha256>      the only copy
  ^                                     ^
  |                                     |
~/.lmstudio/models/unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf   (symlink)
$OLLAMA_MODELS/blobs/sha256-<sha256>                                       (symlink)
$OLLAMA_MODELS/manifests/hf.co/unsloth/Qwen3.6-27B-GGUF/UD-Q4_K_XL         (manifest)
$OLLAMA_MODELS/manifests/registry.ollama.ai/library/qwen3.6/27b            (alias)
```

---

## Installation

Requires Python 3.13+ and [`uv`](https://docs.astral.sh/uv/) or pip.

```bash
git clone https://github.com/StephenGenusa/hf-hub-utilities.git
cd hf-hub-utilities
uv pip install -e .
```

This installs `hfu` and `hf-xfer`, plus the `hf` CLI that `hfu` runs, which
comes with the `huggingface-hub` dependency (2.0 or later, which added the
shared blob store described under [Cache maintenance](#cache-maintenance)).

The tools run on Linux, macOS and Windows. The cache and both views are built
from symlinks, so on Windows turn on Developer Mode (or run as administrator)
to allow creating them. The Ollama service setup below is for Linux.

---

## `hfu` — download into the cache

```
hfu <url_or_path> [-q QUANT ...] [--mmproj MODE] [--revision REV] [--no-sync] [hf download options...]
```

Accepts `https://huggingface.co/<org>/<name>`, `https://hf.co/...`, `org/name`,
and `datasets/…`, `models/…`, `spaces/…` prefixes. Files land in the hub cache
exactly as `hf download` would put them.

For GGUF repos it lists the quants with sizes and prompts for a choice; with
`-q` it skips the prompt. The picker marks quants already in the cache:
`✓ cached`, `partial (n/m)` for a sharded quant with some shards missing, or
`↻ update available` when the Hub now serves different bytes for that file.

The matching `mmproj` file is added for vision models (`--mmproj
F16|BF16|F32|all|none`, default F16 > BF16 > F32), and `README.md` comes along
when present, together with every image, audio and video file in the repo
(`.png`, `.jpg`, `.svg`, `.mp3`, `.wav`, `.mp4`, ...), so the model card's
pictures and samples work offline. Files are requested by exact name, never by
glob. Repos without GGUF files, and datasets and spaces, are downloaded whole.

`--revision` (branch, tag or commit) applies to both the quant listing and the
download, so the picker shows the files that exist at that revision.

```bash
hfu unsloth/Qwen3.6-35B-A3B-GGUF                 # interactive picker
hfu unsloth/Qwen3.6-35B-A3B-GGUF -q UD-Q4_K_S    # scripted
hfu unsloth/Qwen3.6-35B-A3B-GGUF -q Q8_0,UD-Q4_K_S --dry-run
hfu unsloth/Qwen3.6-35B-A3B-GGUF -q Q8_0 --revision v2  # quants as of a branch or tag
hfu https://huggingface.co/zai-org/GLM-OCR       # non-GGUF repo, downloaded whole
```

After a successful download `hfu` runs `hf-xfer sync --execute` so the new
model appears in LM Studio and Ollama; `--no-sync` skips that.

For GGUF repos downloaded before `hfu` fetched assets, run
`hf-xfer cache fetch-assets` (see [Cache maintenance](#cache-maintenance)).

---

## `hf-xfer` — commands

```
hf-xfer sync   [--view lmstudio|ollama] [--execute] [--offline] [--allow-mass-removal]
hf-xfer view status  [--view ...]
hf-xfer view add     <org/name>:<relpath> [--view ...] [--allow-mass-removal] [--execute]
hf-xfer view remove  <org/name>:<relpath> [--view ...] [--execute]
hf-xfer view remove  --foreign <foreign key> [--view ...] [--execute]
hf-xfer view adopt   <foreign key> --repo-id <org/name> [--move] [--execute]
hf-xfer dupes        [--view ...]
hf-xfer verify       [--view ...] [--hub]

hf-xfer cache report
hf-xfer cache repair            [--execute]
hf-xfer cache dedupe            [--execute]
hf-xfer cache prune-superseded  [--execute]
hf-xfer cache thin --min-quant IQ4_XS --max-size 23G [--repo org/name ...] [--allow-empty] [--execute]
hf-xfer cache fetch-assets      [--repo org/name ...] [--execute]
hf-xfer cache gc                [--execute]

hf-xfer import --local-dir <path> [--repo-id ...] [--repo-type ...] [--revision ...] [--cache-dir ...]
               [--move|--hardlink] [--offline] [--no-verify] [--execute]
hf-xfer export --local-dir <path> [--repo-id ...] [--repo-type ...] [--revision ...] [--cache-dir ...]
               [--move|--hardlink] [--execute]
```

### Configuration

`~/.config/hfhub/config.toml` on every platform (on Windows,
`%USERPROFILE%\.config\hfhub\config.toml`), or the path in `$HFHUB_CONFIG`:

```toml
[views.lmstudio]
root = "~/.lmstudio/models"

[views.ollama]
root = "/path/to/huggingface/ollama"      # set OLLAMA_MODELS to the same path

[views.ollama.aliases]
"qwen3.6:27b" = "unsloth/Qwen3.6-27B-GGUF:Qwen3.6-27B-UD-Q4_K_XL.gguf"
"llama3.1:8b" = "unsloth/Llama-3.1-8B-Instruct-GGUF:Llama-3.1-8B-Instruct-Q4_K_M.gguf"
```

A view without a `root` is disabled. View roots must exist before the first
sync; `mkdir -p` them yourself. `view adopt` appends aliases to this file in
place, preserving comments and anything else in it.

---

## Views: LM Studio and Ollama

`hf-xfer sync` reconciles each view with the cache. Every weight GGUF in the
cache gets an entry unless it has been tombstoned.

| View | What sync writes |
|------|------------------|
| `lmstudio` | `<root>/<org>/<name>/<file>.gguf` symlinks to the blob, the chosen mmproj beside them |
| `ollama` | `blobs/sha256-<hash>` symlinks, a manifest under `manifests/hf.co/<org>/<name>/<tag>`, and one alias manifest per configured alias |

Ollama manifests are built from the same registry endpoint `ollama pull
hf.co/...` uses, `https://huggingface.co/v2/<org>/<name>/manifests/<file>`,
so the chat template and parameters match a native pull, while the model
layer always points at the local file. The response and the config blob are
cached under `<root>/.hfhub-registry/`, so later `--offline` runs need no
network. A repo that is not on the Hub gets a minimal manifest and relies on
the chat template embedded in the GGUF.

### Names in Ollama

The canonical name is Ollama's own convention for Hub models:

```
hf.co/unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL
```

The tag is the quant token from the filename (`UD-Q4_K_XL`, `Q8_0`, `f16`) when
it is unique within the repo, otherwise the filename. Names are exact; Ollama
does no partial matching, though it is case-insensitive. Add short names in
the config's alias table; an alias is a second manifest sharing the same
blobs, so it costs nothing:

```bash
ollama run qwen3.6:27b          # alias
ollama run hf.co/unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL   # same model
```

### Rules

- Dry run is the default. Nothing is written without `--execute`.
- Deleting a model inside LM Studio or with `ollama rm` sticks: sync records a
  tombstone and does not recreate it. `view add` brings it back.
- When a cached file is replaced (a newer download of the same file, or an
  mmproj added to the repo), sync relinks the entry: the LM Studio link, the
  Ollama manifest, its alias manifests and blob links all move to the new file,
  and the old blob link is dropped. This applies only to entries sync owns
  whose links still lead into the cache; anything else in their place is
  reported as foreign and left alone.
- Sync only ever deletes inside the view root. A state file listing a path
  that is absolute or climbs out with `..` is refused and that view is
  skipped; a path that leads outside through a symlinked folder is left in
  place with a warning.
- Deleting only a secondary file, an alias manifest, a projector link, an
  mmproj link, is treated as damage and repaired on the next sync. Deleting
  the primary file, the weight link in LM Studio or the `hf.co` manifest or
  model blob link in Ollama, is what records a tombstone.
- Sync never deletes or overwrites anything it did not create, with one
  exception: an Ollama blob that no manifest references any more is removed
  the same way `ollama rm` would remove it. Files that appear in a view
  without HF backing are listed as *foreign* and left alone.
- Sharded GGUFs are not grouped: LM Studio links every shard beside its
  siblings, Ollama skips shards because it cannot load split files.
- Two brakes protect against an absent source of truth: sync aborts if the
  cache directory is missing, and refuses a plan that would remove every entry
  a view owns unless `--allow-mass-removal` is given.
- A registry outage skips the affected entries with a notice; the next sync
  fills them in.

### Foreign files: listing, adopting, deleting

`view status` lists files present in a view that sync does not own: models
LM Studio downloaded itself, or `ollama pull` fetched from Ollama's registry.

```bash
hf-xfer view adopt "lmstudio:OBLITERATUS/Qwen3.8-27B-OBLITERATED/x.gguf" \
        --repo-id OBLITERATUS/Qwen3.8-27B-OBLITERATED --move --execute
```

Adoption moves (or copies) the bytes into the cache. If the file's SHA-256 is
in the repo's current Hub revision, it is filed under that real commit and
etag; otherwise under a synthesised revision, with a note. For an Ollama
registry model the old name is kept as an alias and its template and
parameters are carried into the new manifest. After the move, sync turns the
foreign file into a link.

`hf-xfer dupes` compares foreign files against cache entries by GGUF header,
architecture, quant type, and parameter count, and lists likely matches; it
never acts. `view remove --foreign <key> --execute` deletes one foreign item:
the file for LM Studio, the manifest plus any blob no other manifest references
for Ollama. It is the only command that deletes something sync did not create.

---

## Cache maintenance

These act on the hub cache itself and are dry runs without `--execute`. After
any deleting command, run `hf-xfer sync --execute` so the views drop the
removed entries.

| Command | What it does |
|---------|--------------|
| `cache report` | How much `prune-superseded`, `dedupe` and `repair` would act on, the size of partial downloads (`*.incomplete`), shared-store files no repo uses (for `gc`), and the repos holding 4+ quants of one model (candidates for `thin`) |
| `cache repair` | Recreates snapshot links for blobs nothing references, matching SHA-256 and size against the repo's current Hub revision. Blobs the Hub no longer lists are left alone. |
| `cache dedupe` | Hardlinks identical blobs stored under several repos; turns plain-file copies inside snapshot folders into links (moving the bytes into `blobs/` if needed). When one copy is already in the shared blob store, the others are linked into the store instead. No data is lost. |
| `cache prune-superseded` | Deletes blobs that only an older snapshot links where `refs/main` now links a different blob at the same path. Files that exist only in an old snapshot are kept. |
| `cache thin` | Deletes quants below `--min-quant` or above `--max-size`, in repos where at least one quant survives (`--allow-empty` overrides). MTP draft files and mmproj files are never touched. |
| `cache fetch-assets` | Downloads the image, audio and video files (the pictures and samples a README shows) that cached model repos lack, at the commit already in the cache, so nothing else is updated. For GGUF repos downloaded before `hfu` fetched assets. Dataset repos are skipped: their images are usually the data. |
| `cache gc` | Deletes files in the shared blob store (below) that no repo links any more. |

```bash
hf-xfer cache report
hf-xfer cache repair --execute
hf-xfer cache dedupe --execute
hf-xfer cache prune-superseded --execute
hf-xfer cache thin --min-quant IQ4_XS --max-size 23G --execute
hf-xfer cache fetch-assets --execute
hf-xfer cache gc --execute
hf-xfer sync --execute
```

A dry run reports the space the same command with `--execute` would free. A
blob hardlinked into several repos (after `dedupe`) counts only when every
link to it is deleted, and a shared-store file only when every repo using it
loses it; deleting one of several copies frees nothing.

A note on snapshot folders: the `blobs/` directory holds content under
SHA-256 names only; `snapshots/<commit>/<path>` links give those blobs their
filenames and revisions, and `refs/main` says which snapshot is current.
Every consumer resolves through the links, so a repo whose snapshot folder is
missing looks empty even with every byte present. `cache repair` rebuilds the
links from the Hub listing without moving a byte.

A note on the shared blob store: newer `huggingface_hub` releases keep the
bytes of a Xet-backed download once per cache, in `<hub>/blobs/<xx>/<xet hash>`,
and make the repo's `blobs/<sha256>` a link to it. The repo blob's name is still
the SHA-256; the store file's name is not. Every command here reads the SHA-256
from the repo blob, and the deleting commands (`thin`, `prune-superseded`,
`export --move`) let `huggingface_hub` collect a store file once no repo uses
it. `cache gc` cleans up store files left behind by older versions of this tool
or by deleting repo blobs by hand; `cache report` shows how much that is.

---

## Checking everything: `hf-xfer verify`

`verify` is read-only. It checks the filesystem itself, not sync's state file,
so it also finds damage done outside sync:

- the cache: refs without a snapshot, broken snapshot links, partial
  downloads, shared-store files nothing uses;
- LM Studio: broken links, links to cache files no snapshot uses, real files
  that could be adopted;
- Ollama: manifests pointing at missing or wrongly sized blobs, blobs no
  manifest uses;
- for each view, what `sync --execute` would still change.

With `--hub` it also compares every cached GGUF with the file the Hub's main
revision serves now and lists files that are outdated or were renamed there.
The exit status is 1 when it found a problem, so it can run from cron.

---

## Ollama setup

Ollama runs as its own user, so it needs to reach the cache and write its
store. With the cache on an external drive mounted under `/media/<you>`:

```bash
# let the ollama user traverse the mount point (only that user, nothing else)
sudo setfacl -m u:ollama:x /media/<you>

# the view root, group-writable by ollama, default ACLs so its own subdirs stay writable
mkdir -p "$HF_HOME/ollama"
sudo chgrp ollama "$HF_HOME/ollama" && sudo chmod 2775 "$HF_HOME/ollama"
sudo setfacl -R -m u:<you>:rwx,g:ollama:rwx,d:u:<you>:rwx,d:g:ollama:rwx "$HF_HOME/ollama"

# point the service at it and make it wait for the drive
sudo systemctl edit ollama
#   [Unit]
#   RequiresMountsFor=/media/<you>/<drive>
#   [Service]
#   Environment="OLLAMA_MODELS=/media/<you>/<drive>/huggingface/ollama"
sudo systemctl restart ollama
```

Then `hf-xfer sync --execute` and `ollama list`.

---

## `hf-xfer import` / `export` — local-dir and cache

Hugging Face has two on-disk layouts: the hub cache (content-addressed blobs
plus snapshot links) and the flat `--local-dir` copy. `hf-xfer` translates
between them.

**`import`** seeds the cache from a flat copy, for models that arrived by
USB drive, rsync, or another tool, so `hf download` treats them as cached.
The Hub supplies the commit and per-file etags; `--offline` uses the
`.metadata` files a previous `hf download --local-dir` wrote. Transfer modes:
copy (default), `--move`, `--hardlink` (same filesystem only; falls back to a
copy across devices).

**`export`** materialises a flat copy of a cached repo with metadata so later
`hf download --local-dir` calls are incremental. With `--move`, blobs shared by
several revisions are copied rather than moved, and a file in the shared blob
store is copied out and the repo's link dropped, so the store file is deleted
only once no other repo uses it.

```bash
hf-xfer import --local-dir ~/models/google/gemma-3-4b-it              # dry run
hf-xfer import --local-dir ~/models/google/gemma-3-4b-it --move --execute
hf-xfer export --local-dir ~/export/gemma --repo-id google/gemma-3-4b-it --execute
```

`--repo-id` defaults to the last two path components of `--local-dir`.
`--repo-type` is `model`, `dataset` or `space` (default `model`); `--revision`
is a branch, tag or commit (default `main`). After an import, `hf download`
checks the repo is complete unless `--no-verify` is given. The
cache directory resolves as `--cache-dir` > `HF_HUB_CACHE` >
`HUGGINGFACE_HUB_CACHE` > `HF_HOME/hub` > `~/.cache/huggingface/hub`.

### Moving models to another machine

Copying `models--*` folders directly does not work on most USB drives: the
snapshot files are symlinks, which exFAT and FAT32 cannot store. Export to the
drive, then import on the other machine:

```bash
# this machine
hf-xfer export --local-dir /mnt/usb/hf/unsloth/Qwen3.6-27B-GGUF --repo-id unsloth/Qwen3.6-27B-GGUF --execute
# the other machine; --repo-id comes from the path, --offline if it has no network
hf-xfer import --local-dir /mnt/usb/hf/unsloth/Qwen3.6-27B-GGUF --execute
hf-xfer sync --execute
```

Format the drive as exFAT or NTFS, not FAT32: many GGUF files are over 4 GB.
Pass `--repo-type dataset` on both sides for a dataset.

---

## Development

```bash
uv sync                                                   # locked versions from uv.lock, plus pytest
uv run python -m pytest -q                                # unit tests, no network
HFHUB_LIVE=1 uv run python -m pytest -q -k live           # hits huggingface.co
HFHUB_OLLAMA=1 uv run python -m pytest tests/test_ollama_integration.py -q   # starts a scratch ollama serve
```

`uv.lock` pins the dependency versions the suite was run against; after
changing a dependency in `pyproject.toml`, run `uv lock` and commit both.

The suite passes on Linux and Windows. Code that builds a path inside a repo or
a view uses `/` on every platform (`Path.as_posix()`), because Hub filenames,
view keys, aliases and registry tags all use `/`. Read links with
`blobs.read_link` (also importable as `cache.read_link`), not `os.readlink`,
which on Windows adds a `\\?\` prefix to absolute targets.

Layout, all under `hfhub/`: `cli.py` is the `hf-xfer` command line;
`cache.py` scans the cache; `blobs.py` handles repo blobs in both cache layouts
(regular files and links into the shared store); `config.py` reads and edits the config file;
`state.py` keeps each view's ownership and tombstone record
(`<view root>/.hfhub-state.json`); `views/base.py` holds the pure reconcile
table and the single write path; `views/lmstudio.py` and `views/ollama.py` are
the two views; `ollama_registry.py` talks to the registry endpoint;
`gguf_header.py` reads GGUF metadata; `sync.py` drives the views; `adopt.py`,
`dupes.py`, and `cache_ops.py` are the maintenance commands; `verify.py` is
the read-only `verify` check; `xfer.py`
(import/export) and `hfu.py` are the original tools. `hfu.py` and `hf_xfer.py`
at the repo root are shims for running from a checkout without installing.

## License

MIT. See [LICENSE](LICENSE).
