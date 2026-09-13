from pathlib import Path

import pytest

from hfhub.state import Owned, State
from hfhub.views import base
from hfhub.views.base import Desired, Presence, SkipEntry


def d(key: str, sha: str = "s") -> Desired:
    return Desired(key=key, sha256=sha, links={f"{key}.gguf": Path("/blob")}, extra={})


def kinds(plan):
    return {a.key: a.kind for a in plan.actions}


def test_table_rows():
    desired = {k: d(k) for k in ["new", "adoptable", "ok", "deleted", "tomb", "wrong"]}
    presence = {"new": Presence.ABSENT, "adoptable": Presence.CORRECT, "ok": Presence.CORRECT,
                "deleted": Presence.ABSENT, "tomb": Presence.ABSENT, "wrong": Presence.WRONG}
    state = State(owned={"ok": Owned(["ok.gguf"], "s"), "deleted": Owned(["deleted.gguf"], "s"),
                         "gone": Owned(["gone.gguf"], "s")},
                  tombstones={"tomb": "t"})
    plan = base.reconcile(desired, presence, state)
    assert kinds(plan) == {"new": "create", "adoptable": "adopt", "ok": "noop", "deleted": "tombstone",
                           "tomb": "skip", "wrong": "foreign", "gone": "prune"}


def test_owned_but_wrong_is_reported_not_touched():
    plan = base.reconcile({"k": d("k")}, {"k": Presence.WRONG}, State(owned={"k": Owned(["k.gguf"], "s")}))
    assert kinds(plan) == {"k": "foreign"}


def test_partial_is_repaired_whether_owned_or_not():
    desired = {"fresh": d("fresh"), "owned": d("owned"), "tomb": d("tomb")}
    presence = {k: Presence.PARTIAL for k in desired}
    state = State(owned={"owned": Owned(["owned.gguf"], "s")}, tombstones={"tomb": "t"})
    plan = base.reconcile(desired, presence, state)
    assert kinds(plan) == {"fresh": "create", "owned": "create", "tomb": "skip"}


class FakeView:
    name = "fake"
    root = Path("/view")

    def __init__(self):
        self.created: list[str] = []
        self.removed: list[str] = []
        self.fail: set[str] = set()

    def desired(self, entries):
        return {}

    def present(self, d):
        return Presence.ABSENT

    def create(self, d):
        if d.key in self.fail:
            raise SkipEntry("offline")
        self.created.append(d.key)
        return list(d.paths) + ["shared/small"]

    def remove(self, paths):
        self.removed += paths

    def foreign(self, state):
        return []


def test_apply_dry_run_writes_nothing():
    v = FakeView()
    plan = base.reconcile({"k": d("k")}, {"k": Presence.ABSENT}, State())
    state = base.apply(plan, v, State(), execute=False)
    assert v.created == [] and state.owned == {}


def test_apply_execute_creates_and_records_paths():
    v = FakeView()
    desired = {"k": d("k")}
    plan = base.reconcile(desired, {"k": Presence.ABSENT}, State())
    state = base.apply(plan, v, State(), execute=True, desired=desired)
    assert v.created == ["k"]
    assert state.owned["k"].paths == ["k.gguf", "shared/small"]


def test_apply_prune_keeps_paths_owned_by_others():
    v = FakeView()
    state = State(owned={"a": Owned(["a.gguf", "shared/small"], "s"), "b": Owned(["b.gguf", "shared/small"], "s")})
    desired = {"b": d("b")}
    plan = base.reconcile(desired, {"b": Presence.CORRECT}, state)
    state = base.apply(plan, v, state, execute=True, desired=desired)
    assert v.removed == ["a.gguf"]
    assert "a" not in state.owned and "b" in state.owned


def test_apply_tombstone_records_timestamp():
    v = FakeView()
    state = State(owned={"k": Owned(["k.gguf"], "s")})
    desired = {"k": d("k")}
    plan = base.reconcile(desired, {"k": Presence.ABSENT}, state)
    state = base.apply(plan, v, state, execute=True, desired=desired)
    assert "k" not in state.owned and "k" in state.tombstones


def test_apply_skip_on_create_failure_leaves_state_clean():
    v = FakeView()
    v.fail.add("k")
    desired = {"k": d("k")}
    plan = base.reconcile(desired, {"k": Presence.ABSENT}, State())
    state = base.apply(plan, v, State(), execute=True, desired=desired)
    assert state.owned == {}
    assert [a.kind for a in plan.actions] == ["skip"]


def test_apply_create_without_desired_raises():
    v = FakeView()
    plan = base.reconcile({"k": d("k")}, {"k": Presence.ABSENT}, State())
    with pytest.raises(ValueError):
        base.apply(plan, v, State(), execute=True)


def test_apply_repair_merges_paths_into_existing_owned():
    v = FakeView()
    state = State(owned={"k": Owned(["k.gguf", "stale/extra"], "s")})
    desired = {"k": d("k")}
    plan = base.reconcile(desired, {"k": Presence.PARTIAL}, state)
    state = base.apply(plan, v, state, execute=True, desired=desired)
    assert v.created == ["k"]
    assert state.owned["k"].paths == ["k.gguf", "shared/small", "stale/extra"]
