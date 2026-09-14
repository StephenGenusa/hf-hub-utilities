from hfhub import cli


def test_import_export_delegate(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli._xfer, "main", lambda argv: seen.update(argv=argv) or 0)
    assert cli.xfer_main(["import", "--local-dir", "/x"]) == 0
    assert seen["argv"] == ["import", "--local-dir", "/x"]


def test_sync_parses_flags(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.sync, "run", lambda config, views, execute, offline, out, allow_mass_removal:
                        seen.update(views=views, execute=execute, offline=offline,
                                    allow_mass_removal=allow_mass_removal) or {})
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(path=None, views={}))
    assert cli.xfer_main(["sync", "--view", "ollama", "--execute", "--offline"]) == 0
    assert seen == {"views": ["ollama"], "execute": True, "offline": True, "allow_mass_removal": False}
    assert cli.xfer_main(["sync", "--allow-mass-removal"]) == 0
    assert seen["allow_mass_removal"] is True


def test_view_status_default_all_views(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.sync, "status", lambda config, views, out: seen.update(views=views))
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(path=None, views={}))
    assert cli.xfer_main(["view", "status"]) == 0
    assert seen["views"] == ["lmstudio", "ollama"]


def test_hfu_no_sync_flag_is_own_option():
    from hfhub.hfu import _split_own_args
    own, extra = _split_own_args(["org/name", "--no-sync", "--revision", "main"])
    assert "--no-sync" in own and extra == ["--revision", "main"]


def test_help_lists_import_and_export():
    help_text = cli._parser().format_help()
    assert "import" in help_text and "export" in help_text


def test_view_add_threads_allow_mass_removal(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.sync, "view_add", lambda config, key, views, execute, out, allow_mass_removal:
                        seen.update(key=key, allow_mass_removal=allow_mass_removal))
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(path=None, views={}))
    assert cli.xfer_main(["view", "add", "o/r:f.gguf", "--execute"]) == 0
    assert seen == {"key": "o/r:f.gguf", "allow_mass_removal": False}
    assert cli.xfer_main(["view", "add", "o/r:f.gguf", "--allow-mass-removal"]) == 0
    assert seen["allow_mass_removal"] is True


def test_cache_thin_parses_rules(monkeypatch, tmp_path):
    from hfhub import cache_ops
    seen = {}
    monkeypatch.setattr(cli.sync, "hub_dir", lambda: tmp_path)
    monkeypatch.setattr(cache_ops, "thin", lambda hub, execute, min_bits, max_size, repos, allow_empty, out: seen.update(
        hub=hub, execute=execute, min_bits=min_bits, max_size=max_size, repos=repos, allow_empty=allow_empty))
    assert cli.xfer_main(["cache", "thin", "--min-quant", "IQ4_XS", "--max-size", "23G", "--repo", "o/r", "--execute"]) == 0
    assert seen == {"hub": tmp_path, "execute": True, "min_bits": 4, "max_size": 23_000_000_000, "repos": ["o/r"], "allow_empty": False}


def test_cache_thin_requires_a_rule(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.sync, "hub_dir", lambda: tmp_path)
    assert cli.xfer_main(["cache", "thin"]) == 2


def test_cache_commands_dispatch_and_default_to_dry_run(monkeypatch, tmp_path):
    from hfhub import cache_ops
    calls = []
    monkeypatch.setattr(cli.sync, "hub_dir", lambda: tmp_path)
    for name in ("repair", "dedupe", "prune_superseded"):
        monkeypatch.setattr(cache_ops, name, lambda hub, execute, out, _n=name: calls.append((_n, execute)))
    monkeypatch.setattr(cache_ops, "report", lambda hub, out: calls.append(("report", None)))
    for sub in ("report", "repair", "dedupe", "prune-superseded"):
        assert cli.xfer_main(["cache", sub]) == 0
    assert calls == [("report", None), ("repair", False), ("dedupe", False), ("prune_superseded", False)]


def test_cache_commands_refuse_missing_hub(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.sync, "hub_dir", lambda: tmp_path / "nope")
    assert cli.xfer_main(["cache", "report"]) == 2
