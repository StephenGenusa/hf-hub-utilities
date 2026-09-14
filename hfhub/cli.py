"""Console entry points for hfu and hf-xfer."""
from __future__ import annotations

import argparse
import sys

from hfhub import config as cfg, sync
from hfhub import hfu as _hfu
from hfhub import xfer as _xfer

ALL_VIEWS = list(cfg.VIEW_NAMES)


def hfu_main() -> None:
    _hfu.main()


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="hf-xfer",
        description="HF cache <-> local-dir transfer, consumer views (sync/view/dupes), and cache maintenance.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def view_opt(p):
        p.add_argument("--view", action="append", choices=ALL_VIEWS, help="limit to one view (repeatable)")

    def execute_opt(p, what="write changes"):
        p.add_argument("--execute", action="store_true", help=f"{what} (default: dry run)")

    s = sub.add_parser("sync", help="reconcile LM Studio / Ollama views with the HF cache")
    view_opt(s)
    execute_opt(s)
    s.add_argument("--offline", action="store_true", help="never contact huggingface.co")
    s.add_argument("--allow-mass-removal", action="store_true",
                   help="apply a plan that removes every entry sync owns in a view")

    v = sub.add_parser("view", help="inspect or edit view entries")
    vs = v.add_subparsers(dest="vcmd", required=True)
    p = vs.add_parser("status", help="owned / tombstoned / foreign per view, unlinked cache blobs"); view_opt(p)
    for name, help_ in (("add", "create an entry, clearing its tombstone"), ("remove", "remove an entry and tombstone it")):
        p = vs.add_parser(name, help=help_)
        p.add_argument("key", help="<org/name>:<relpath> or, with --foreign, a foreign key")
        view_opt(p)
        execute_opt(p)
        if name == "add":
            p.add_argument("--allow-mass-removal", action="store_true",
                           help="apply a plan that removes every entry sync owns in a view")
        if name == "remove":
            p.add_argument("--foreign", action="store_true", help="delete a foreign (non-owned) item")
    p = vs.add_parser("adopt", help="move or copy a foreign file into the HF cache, then sync")
    p.add_argument("key", help="foreign key as printed by `view status`")
    p.add_argument("--repo-id", required=True, help="<org/name> to file it under")
    view_opt(p)
    p.add_argument("--move", action="store_true", help="move instead of copy")
    execute_opt(p)

    d = sub.add_parser("dupes", help="list foreign files that look like cache entries")
    view_opt(d)

    c = sub.add_parser("cache", help="maintain the HF cache itself (report/repair/dedupe/prune-superseded/thin)")
    cs = c.add_subparsers(dest="ccmd", required=True)
    cs.add_parser("report", help="how much each maintenance command could reclaim")
    p = cs.add_parser("repair", help="recreate snapshot links for blobs that nothing references"); execute_opt(p, "create links")
    p = cs.add_parser("dedupe", help="hardlink identical blobs across repos; turn plain copies in snapshots into links"); execute_opt(p)
    p = cs.add_parser("prune-superseded", help="delete old-revision blobs replaced by a newer file at the same path"); execute_opt(p, "delete")
    p = cs.add_parser("thin", help="delete quants below a bit width or above a size where another quant survives")
    p.add_argument("--min-quant", metavar="QUANT", help="drop quants below this, e.g. IQ4_XS or 4")
    p.add_argument("--max-size", metavar="SIZE", help="drop files larger than this, e.g. 23G")
    p.add_argument("--repo", action="append", metavar="ORG/NAME", help="limit to a repo (repeatable)")
    p.add_argument("--allow-empty", action="store_true", help="allow a repo to lose its last quant")
    execute_opt(p, "delete")

    # Intercepted in xfer_main before argparse ever sees them; they are here so
    # that `hf-xfer --help` lists every command the tool actually has.
    sub.add_parser("import", add_help=False, help="local-dir -> cache (run `hf-xfer import --help`)")
    sub.add_parser("export", add_help=False, help="cache -> local-dir (run `hf-xfer export --help`)")
    return ap


def _cache_cmd(args) -> int:
    from hfhub import cache_ops
    hub = sync.hub_dir()
    if not hub.is_dir():
        print(f"[cache] not found: {hub}")
        return 2
    if args.ccmd == "report":
        cache_ops.report(hub, out=print)
    elif args.ccmd == "repair":
        cache_ops.repair(hub, execute=args.execute, out=print)
    elif args.ccmd == "dedupe":
        cache_ops.dedupe(hub, execute=args.execute, out=print)
    elif args.ccmd == "prune-superseded":
        cache_ops.prune_superseded(hub, execute=args.execute, out=print)
    elif args.ccmd == "thin":
        if not args.min_quant and not args.max_size:
            print("thin: give --min-quant and/or --max-size")
            return 2
        cache_ops.thin(hub, execute=args.execute,
                       min_bits=cache_ops.parse_min_quant(args.min_quant) if args.min_quant else None,
                       max_size=cache_ops.parse_size(args.max_size) if args.max_size else None,
                       repos=args.repo, allow_empty=args.allow_empty, out=print)
    return 0


def xfer_main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("import", "export"):
        return _xfer.main(argv)
    args = _parser().parse_args(argv)
    if args.cmd == "cache":
        return _cache_cmd(args)
    config = cfg.load()
    views = args.view or ALL_VIEWS
    if args.cmd == "sync":
        sync.run(config, views, execute=args.execute, offline=args.offline, out=print,
                 allow_mass_removal=args.allow_mass_removal)
        return 0
    if args.cmd == "view":
        if args.vcmd == "status":
            sync.status(config, views, out=print)
        elif args.vcmd == "add":
            sync.view_add(config, args.key, views, execute=args.execute, out=print,
                          allow_mass_removal=args.allow_mass_removal)
        elif args.vcmd == "remove":
            if args.foreign:
                from hfhub import adopt
                adopt.remove_foreign(config, args.key, views, execute=args.execute)
            else:
                sync.view_remove(config, args.key, views, execute=args.execute, out=print)
        elif args.vcmd == "adopt":
            from hfhub import adopt
            adopt.run(config, args.key, args.repo_id, views, move=args.move, execute=args.execute)
        return 0
    if args.cmd == "dupes":
        from hfhub import dupes
        dupes.run(config, views)
        return 0
    return 2
