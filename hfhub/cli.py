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
    ap = argparse.ArgumentParser(prog="hf-xfer", description="HF cache <-> local-dir transfer, and consumer views (sync/view/dupes).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def view_opt(p):
        p.add_argument("--view", action="append", choices=ALL_VIEWS, help="limit to one view (repeatable)")

    s = sub.add_parser("sync", help="reconcile LM Studio / Ollama views with the HF cache")
    view_opt(s)
    s.add_argument("--execute", action="store_true", help="write changes (default: dry run)")
    s.add_argument("--offline", action="store_true", help="never contact huggingface.co")
    s.add_argument("--allow-mass-removal", action="store_true",
                   help="apply a plan that removes every entry sync owns in a view")

    v = sub.add_parser("view", help="inspect or edit view entries")
    vs = v.add_subparsers(dest="vcmd", required=True)
    p = vs.add_parser("status"); view_opt(p)
    for name in ("add", "remove"):
        p = vs.add_parser(name)
        p.add_argument("key", help="<org/name>:<relpath> or, with --foreign, a foreign key")
        view_opt(p)
        p.add_argument("--execute", action="store_true")
        if name == "add":
            p.add_argument("--allow-mass-removal", action="store_true",
                           help="apply a plan that removes every entry sync owns in a view")
        if name == "remove":
            p.add_argument("--foreign", action="store_true", help="delete a foreign (non-owned) item")
    p = vs.add_parser("adopt")
    p.add_argument("key")
    p.add_argument("--repo-id", required=True)
    view_opt(p)
    p.add_argument("--move", action="store_true")
    p.add_argument("--execute", action="store_true")

    d = sub.add_parser("dupes", help="list foreign files that look like cache entries")
    view_opt(d)

    # Intercepted in xfer_main before argparse ever sees them; they are here so
    # that `hf-xfer --help` lists every command the tool actually has.
    sub.add_parser("import", add_help=False, help="local-dir -> cache (run `hf-xfer import --help`)")
    sub.add_parser("export", add_help=False, help="cache -> local-dir (run `hf-xfer export --help`)")
    return ap


def xfer_main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("import", "export"):
        return _xfer.main(argv)
    args = _parser().parse_args(argv)
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
