# Consumer views: LM Studio and Ollama on top of the HF hub cache

**Date:** 2026-09-13
**Status:** approved design, pending implementation plan

## Goal

Keep the Hugging Face hub cache as the single store of GGUF model bytes, and
maintain two derived *views* of it so that LM Studio and Ollama both see every
cached model without a second copy on disk:

- **LM Studio view** – a flat `org/name/file.gguf` symlink tree under LM Studio's
  models folder.
- **Ollama view** – `sha256-<hash>` blob symlinks plus hand-written manifests
  under an `OLLAMA_MODELS` directory, with optional short-name aliases.

The HF cache is the only source of truth. Views are regenerated from it. Files
that appear in a view without HF backing are reported, never imported
automatically; `hf-xfer import` remains the explicit path for that.

## Decisions (settled with the user)

| # | Decision |
|---|---|
| 1 | HF cache is the source of truth. Sync is one-way. Orphans are reported; `hf-xfer import` is the explicit way to adopt them. |
| 2 | Ollama models are named by Ollama's own convention `hf.co/<org>/<name>:<tag>`, plus optional short aliases from config (`ollama cp` semantics: a second manifest, zero extra bytes). |
| 3 | Sync writes Ollama manifests itself. Template/params/config layers are fetched over plain HTTP from `https://huggingface.co/v2/<org>/<name>/manifests/<tag>` and `.../blobs/<digest>` when the repo exists on the Hub; the model (and projector) digest is swapped for the local hash. Non-Hub repos get a minimal manifest and a warning that the model relies on its embedded chat template. |
| 4 | Sync records what it created in a per-view state file. An owned entry that has disappeared from disk is a deliberate user deletion: it is tombstoned and not recreated. `view add` clears a tombstone. |
| 5 | Every weight GGUF in the cache that is not tombstoned gets a view entry in both views. |
| 6 | The Ollama store lives at `$HF_HOME/ollama`, owned by the user, group `ollama`, setgid + group-writable, so sync writes as the user and Ollama's own pulls write as `ollama`. One-time system steps: `chmod o+x /media/stephen`, unit override with `OLLAMA_MODELS` and `RequiresMountsFor`. |
| 7 | Sync runs manually via `hf-xfer sync`, and automatically at the end of a successful `hfu` download unless `--no-sync` is given. |
| 8 | Code is restructured into a small package; console script names stay `hfu` and `hf-xfer`. |

## Verified compatibility (tests run 2026-09-13, Ollama 0.30.6)

- HF cache: blob filename is the file's SHA-256; snapshot entries are relative
  symlinks `../../blobs/<hash>`; `refs/main` holds the commit.
- Ollama: a symlink `blobs/sha256-<hash>` pointing at an HF blob is accepted by
  `ollama pull hf.co/…` (download skipped, only stat + size used), `ollama run`
  (text and vision via projector layer) and `ollama rm` (removes only the
  symlink). A hand-written manifest under `manifests/hf.co/<org>/<name>/<tag>`
  appears in `ollama list` immediately and runs.
- `ollama create FROM <path>` re-serialises the GGUF into a new blob with a
  different hash; it is **not** used.
- `https://huggingface.co/v2/<org>/<name>/manifests/<tag>` returns the manifest
  without auth for public repos; `<tag>` may be the quant name or the full
  filename; template and params blobs are served from `…/blobs/<digest>`;
  llama-family repos receive a Go template layer.
- LM Studio: already loads 137 absolute symlinks into HF blobs; mmproj is
  paired by folder.

## Package layout

```
hfhub/
  __init__.py
  cache.py            scan HF cache -> list[GgufEntry]
  config.py           ~/.config/hfhub/config.toml -> Config
  state.py            per-view state file (owned + tombstones)
  views/__init__.py
  views/base.py       View protocol, Plan, reconcile()
  views/lmstudio.py   LmStudioView
  views/ollama.py     OllamaView (blob links, manifests, aliases)
  ollama_registry.py  HTTP client for huggingface.co/v2 manifests + blobs
  gguf_header.py      stdlib GGUF header reader (KV + tensor info, no tensor data)
  adopt.py            foreign -> HF cache import (Hub-verified or synthesised)
  dupes.py            foreign vs cache matching by header
  xfer.py             existing import/export, moved verbatim
  hfu.py              existing downloader + post-download sync hook
  cli.py              argparse wiring for `hfu` and `hf-xfer`
tests/
  test_hfu.py         existing (moved)
  test_cache.py, test_state.py, test_reconcile.py,
  test_lmstudio.py, test_ollama.py, test_ollama_registry.py, test_cli.py,
  test_gguf_header.py, test_adopt.py, test_dupes.py
```

`pyproject.toml`: `[tool.setuptools] packages = ["hfhub", "hfhub.views"]`,
scripts `hfu = "hfhub.cli:hfu_main"`, `hf-xfer = "hfhub.cli:xfer_main"`.
Top-level `hfu.py` and `hf_xfer.py` become two-line shims importing from the
package so `python hfu.py` keeps working.

## Commands

```
hf-xfer sync   [--view lmstudio|ollama] [--execute] [--offline]
hf-xfer view add    <org/name>[:<relpath>] [--view …] [--execute]
hf-xfer view remove <org/name>[:<relpath>] [--view …] [--execute]
hf-xfer view status [--view …]
hf-xfer view adopt  <key> --repo-id <org/name> [--view …] [--move] [--execute]
hf-xfer view remove --foreign <key> [--view …] [--execute]
hf-xfer dupes [--view …]
hfu … [--no-sync]
```

- Dry run is the default for every writing command, as for import/export.
- `sync` is idempotent: a second run immediately after the first plans zero
  actions.
- `--offline` skips the registry fetch; models whose small layers are not yet
  cached are skipped with a notice rather than written with a bare manifest.
- `view status` prints, per view: owned, missing (tombstoned), unlinked (cache
  blob with no snapshot link), foreign (present in the view, not owned).

## Config

`~/.config/hfhub/config.toml` (override with `HFHUB_CONFIG`):

```toml
[views.lmstudio]
root = "~/.lmstudio/models"

[views.ollama]
root = "/media/stephen/2f5f6aa6-3182-4426-b774-5829d6ba8260/huggingface/ollama"

[views.ollama.aliases]
"qwen3.6:27b" = "unsloth/Qwen3.6-27B-GGUF:Qwen3.6-27B-UD-Q4_K_XL.gguf"
```

A view with no `root` is disabled. Missing config file means both views
disabled and `sync` explains how to create one.

## cache.py

`scan(hub_dir) -> list[GgufEntry]`

```python
@dataclass(frozen=True)
class GgufEntry:
    repo_id: str        # "unsloth/Qwen3.6-27B-GGUF"
    relpath: str        # "Qwen3.6-27B-UD-Q4_K_XL.gguf" (may contain '/')
    blob: Path          # absolute path to blobs/<sha256>
    sha256: str         # from blob filename, never re-hashed
    size: int
    is_mmproj: bool
    is_current: bool    # snapshot == refs/main
```

Rules:

- Walk `models--*/snapshots/*/**/*.gguf`. One entry per `(repo_id, relpath)`;
  when several snapshots link the same relpath, the entry from the `refs/main`
  snapshot wins, else the newest snapshot directory.
- Blobs with no snapshot link are returned separately by
  `unlinked(hub_dir) -> list[(repo_id, blob)]` for `view status`.
- Sharded quants are not grouped: each shard is its own entry with
  `is_shard = True`. The LM Studio view links every shard beside its siblings
  (deleting one shard tombstones only that shard); the Ollama view skips shards
  with a warning because Ollama does not load split GGUFs.
- `mmproj_for(entry, entries) -> GgufEntry | None` uses `hfu.pick_mmproj`
  (F16 > BF16 > F32) among mmproj entries of the same repo.
- Read-only. Nothing in this module writes.

## state.py

`<view root>/.hfhub-state.json`

```json
{
  "version": 1,
  "owned": {
    "<repo_id>:<relpath>": {"paths": ["rel/path/in/view", …], "sha256": "…"}
  },
  "tombstones": {"<repo_id>:<relpath>": "2026-09-13T12:04:00"}
}
```

Atomic write (temp file + rename). Missing file = empty state. Paths are
relative to the view root.

## views/base.py

```python
class View(Protocol):
    name: str
    root: Path
    def desired(self, entries: list[GgufEntry]) -> dict[str, Desired]   # key -> what should exist
    def present(self, key: str, desired: Desired) -> bool               # all paths exist and point correctly
    def create(self, desired: Desired) -> list[Path]                    # returns created paths
    def remove(self, paths: list[Path]) -> None
    def foreign(self) -> list[Path]                                     # files in view not in state
```

`reconcile(view, entries, state) -> Plan` implements this table, pure and
unit-tested:

| In cache | Owned | Present | Tombstoned | Action |
|---|---|---|---|---|
| yes | no | no | no | create, mark owned |
| yes | no | yes (correct target) | no | adopt: mark owned, no write |
| yes | yes | yes | no | nothing |
| yes | yes | no | no | tombstone, drop from owned |
| yes | any | any | yes | nothing |
| no | yes | any | any | prune owned paths, drop from owned |
| yes | no | yes (wrong target) | no | foreign: report, never touch |

`apply(plan, view, state, execute)` is the only code path that writes to a
view. Prune removes only paths listed in `owned`, then removes directories
that became empty and are below the view root.

## views/lmstudio.py

- Desired paths for a weight entry: `<root>/<org>/<name>/<basename(relpath)>`
  as an absolute symlink to the blob, plus the chosen mmproj beside it under
  its own basename.
- Shards: one link per shard, same folder.
- Two relpaths in one repo with the same basename (e.g. `MTP/x.gguf` and
  `x.gguf`) collide; the current-snapshot one wins and the other is reported.
- Foreign detection: any `*.gguf` under root not in `owned`. Non-GGUF files
  (`config.json`) are ignored entirely.

## views/ollama.py

Store layout (identical to Ollama's own):

```
<root>/blobs/sha256-<hash>                 symlink -> HF blob (model, projector)
<root>/blobs/sha256-<hash>                 real small files (config, template, params)
<root>/manifests/hf.co/<org>/<name>/<tag>  manifest JSON
<root>/manifests/<alias-namespace>/…       alias manifests
```

- Tag = quant name from `hfu.group_quants` when unique within the repo, else
  the full filename. Tags are validated against Ollama's `[A-Za-z0-9._-]`.
- Manifest assembly:
  1. Fetch `manifests/<filename>` from the registry (`ollama_registry.py`) and
     cache the response under `<root>/.hfhub-registry/<org>/<name>/<filename>.json`.
  2. Download config, template, params blobs into `<root>/blobs/` as real files
     (they are a few hundred bytes).
  3. Replace the `model` layer digest/size with the local blob; replace or add
     the `projector` layer with the local mmproj; rewrite `rootfs.diff_ids` in
     the config blob accordingly and re-hash it.
  4. Write the manifest.
- Registry 404 (repo not on Hub, or file not in current revision and no cached
  response): write a manifest with config + model (+ projector) only. Config is
  synthesised from the GGUF header (`general.architecture`, `general.file_type`,
  parameter count). Emit a warning naming the model.
- Aliases: for each `alias -> "<repo_id>:<relpath>"` in config, write a second
  manifest with identical layers at `manifests/registry.ollama.ai/library/<name>/<tag>`
  (or `<ns>/<name>/<tag>` if the alias has a namespace). Alias manifests are
  owned paths of the target entry, so they are pruned with it.
- Ollama's own pulls from registry.ollama.ai land in the same store as real
  files and are reported as foreign, never touched.
- Blob symlink names must not collide with real files: if
  `sha256-<hash>` already exists as a regular file (Ollama downloaded the same
  bytes), adopt without replacing.

## ollama_registry.py

- `fetch_manifest(org, name, tag) -> dict | None` (None on 404).
- `fetch_blob(org, name, digest) -> bytes`.
- Follows redirects, 30 s timeout, no auth by default; sends the HF token if
  present so private repos work.
- Pure `patch_manifest(manifest, model, projector) -> (manifest, config_bytes)`
  for the digest swap, unit-tested with fixtures captured from the live
  endpoint.

## hfu hook

After a successful `hf download`, hfu calls `sync` for both views in execute
mode, printing its summary. `--no-sync` skips it. A sync failure is reported
but does not change hfu's exit code, since the download itself succeeded.

## Error handling

- Missing view root: `sync` skips that view with a one-line notice.
- Permission denied writing a view: abort that view, report, continue with
  the other.
- Registry unreachable and `--offline` not given: models with no cached
  registry response are skipped with a notice; everything else proceeds.
- State file corrupt: refuse to run against that view until the user moves
  it aside. Never silently start from empty state on a populated view.

## Testing

- All reconcile logic, tag derivation, manifest patching, config synthesis,
  and path derivation are pure functions tested with pytest and `tmp_path`
  fixtures that build a fake HF cache.
- Registry client is tested against captured JSON fixtures; one opt-in live
  test (`HFHUB_LIVE=1`) hits the real endpoint.
- One opt-in integration test (`HFHUB_OLLAMA=1`) starts a scratch
  `ollama serve` on a spare port with `OLLAMA_MODELS` pointing at a temp view
  and asserts `ollama list` shows the synced model and its alias.

## Migration (one-time, after implementation)

1. Create `$HF_HOME/ollama`, `chgrp ollama`, `chmod 2775`, default ACL for
   group write; `sudo chmod o+x /media/stephen`; unit override with
   `OLLAMA_MODELS` and `RequiresMountsFor`; restart Ollama.
2. Repair the 12 unlinked repos (`hf-xfer import`-style symlink recreation
   from the Hub tree; blobs already verified complete).
3. `hf-xfer sync --execute` for both views; thin the LM Studio and Ollama
   lists by deleting quants that are never used (deletions stick).
4. Add aliases for the model names in use today; confirm each runs.
5. Import the two Ollama-only models into the HF cache via `hf-xfer import
   --offline` with synthesised metadata; sync.
6. Download the 11 missing models via `hfu`; sync happens automatically.
7. Delete the old Ollama store on `/`.

## Foreign files: listing, adoption, duplicates

**Listing.** `view status` prints every foreign item with a stable key:

- LM Studio: `lmstudio:<org>/<name>/<basename>` for each real `*.gguf` file
  (symlinks that point outside the HF cache are also foreign).
- Ollama: `ollama:<registry>/<ns>/<name>:<tag>` for each manifest whose model
  layer is a real file (typically `registry.ollama.ai/library/...`).

Each row shows size, GGUF `general.architecture`, parameter count, and
`general.file_type` name read from the header (module `gguf_header.py`,
stdlib-only reader of the KV section; never reads tensor data).

**Adoption.** `view adopt <key> --repo-id <org/name> [--move]` moves or copies
the bytes into the HF cache and re-syncs so the foreign file becomes a link:

1. Query the Hub for `<org/name>`. If it exists and a file with the same
   SHA-256 is in the current revision, import under that filename with the
   real commit and etag, exactly as `hf-xfer import` does.
2. Otherwise (repo missing, or hash not in the repo) synthesise: commit =
   40-hex derived from `sha256(repo_id + sha256)[:40]`, etag = the file
   SHA-256, filename = the foreign basename (for Ollama: `<name>-<tag>.gguf`).
   Print a warning that the entry is not Hub-backed.
3. Write blob, snapshot symlink, `refs/main` via the existing `execute_import`
   machinery. `--move` renames when on the same filesystem, else copies then
   unlinks; without `--move`, copy.
4. For an Ollama registry model, add an alias `<name>:<tag>` to the config so
   the old name keeps working, and carry over its template/params layers into
   the new `hf.co`-style manifest.
5. Run `sync` for the view.

**Duplicates.** `hf-xfer dupes` matches each foreign item against cache
entries using the GGUF header: same `general.architecture`, same
`general.file_type`, and parameter count within 2 %. Parameter count is the
sum of tensor element counts from the header's tensor-info section, which
requires reading only the header. Matches are printed as
`<foreign key>  ≈  <repo_id>:<relpath>  (<reason>)`, never acted on.
`view remove --foreign <key>` deletes the named foreign item: for LM Studio
the file (and its empty folder); for Ollama the manifest plus any blob no
other manifest references. This is the only path that deletes something the
manager did not create, and it requires the key to be spelled out.

## Out of scope

- Importing foreign files automatically; adoption is always an explicit command.
- Watching the cache with a daemon.
- Any change to the import/export behaviour of `hf-xfer`.
