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
