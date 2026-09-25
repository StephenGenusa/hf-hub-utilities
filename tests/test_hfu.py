"""Tests for hfu.py — the pure, non-interactive parts."""
import pytest

from hfhub import hfu
from hfhub.hfu import RepoFile

GB = 1024 ** 3

UNSLOTH_FILES = [
    RepoFile(".gitattributes", 1500),
    RepoFile("BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf", 40 * GB),
    RepoFile("BF16/Qwen3.6-35B-A3B-BF16-00002-of-00002.gguf", 30 * GB),
    RepoFile("Qwen3.6-35B-A3B-MXFP4_MOE.gguf", 20 * GB),
    RepoFile("Qwen3.6-35B-A3B-Q8_0.gguf", 37 * GB),
    RepoFile("Qwen3.6-35B-A3B-UD-Q4_K_S.gguf", 20 * GB),
    RepoFile("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", 22 * GB),
    RepoFile("README.md", 5000),
    RepoFile("imatrix_unsloth.gguf_file", 2 * GB),
    RepoFile("mmproj-BF16.gguf", 800_000_000),
    RepoFile("mmproj-F16.gguf", 800_000_000),
    RepoFile("mmproj-F32.gguf", 1_600_000_000),
]


class TestGroupQuants:
    def test_root_level_files_become_one_quant_each(self):
        names = {q.name for q in hfu.group_quants(UNSLOTH_FILES)}
        assert {"MXFP4_MOE", "Q8_0", "UD-Q4_K_S", "UD-Q4_K_XL"} <= names

    def test_sharded_quant_in_subfolder_is_one_row_with_summed_size(self):
        by_name = {q.name: q for q in hfu.group_quants(UNSLOTH_FILES)}
        bf16 = by_name["BF16"]
        assert bf16.files == [
            "BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf",
            "BF16/Qwen3.6-35B-A3B-BF16-00002-of-00002.gguf",
        ]
        assert bf16.size == 70 * GB

    def test_mmproj_and_non_gguf_files_are_not_quants(self):
        names = {q.name for q in hfu.group_quants(UNSLOTH_FILES)}
        assert not any("mmproj" in n.lower() for n in names)
        assert not any("imatrix" in n.lower() for n in names)
        assert "README" not in names

    def test_prefix_trimming_never_splits_inside_a_quant_token(self):
        files = [
            RepoFile("Llama-3-8B-Q4_K_M.gguf", 5 * GB),
            RepoFile("Llama-3-8B-Q4_K_S.gguf", 4 * GB),
        ]
        assert [q.name for q in hfu.group_quants(files)] == ["Q4_K_S", "Q4_K_M"]

    def test_thebloke_dot_separated_layout(self):
        files = [
            RepoFile("llama-2-7b.Q4_K_M.gguf", 5 * GB),
            RepoFile("llama-2-7b.Q8_0.gguf", 7 * GB),
        ]
        assert [q.name for q in hfu.group_quants(files)] == ["Q4_K_M", "Q8_0"]

    def test_bartowski_folder_layout_with_split_shards(self):
        files = [
            RepoFile("Qwen_Qwen3-32B-Q4_K_M.gguf", 20 * GB),
            RepoFile("Qwen_Qwen3-32B-Q8_0/Qwen_Qwen3-32B-Q8_0-00001-of-00002.gguf", 20 * GB),
            RepoFile("Qwen_Qwen3-32B-Q8_0/Qwen_Qwen3-32B-Q8_0-00002-of-00002.gguf", 15 * GB),
        ]
        by_name = {q.name: q for q in hfu.group_quants(files)}
        assert set(by_name) == {"Q4_K_M", "Q8_0"}
        assert len(by_name["Q8_0"].files) == 2

    def test_single_gguf_repo_uses_full_stem_as_name(self):
        files = [RepoFile("model-f16.gguf", 3 * GB), RepoFile("README.md", 10)]
        [q] = hfu.group_quants(files)
        assert q.name == "model-f16"
        assert q.files == ["model-f16.gguf"]

    def test_sorted_by_size_ascending(self):
        sizes = [q.size for q in hfu.group_quants(UNSLOTH_FILES)]
        assert sizes == sorted(sizes)

    def test_no_gguf_files_gives_empty_list(self):
        assert hfu.group_quants([RepoFile("config.json", 1), RepoFile("model.safetensors", 9)]) == []


MMPROJ = ["mmproj-BF16.gguf", "mmproj-F16.gguf", "mmproj-F32.gguf"]


class TestPickMmproj:
    def test_auto_prefers_f16(self):
        assert hfu.pick_mmproj(MMPROJ, None) == ["mmproj-F16.gguf"]

    def test_auto_falls_back_to_bf16_then_f32(self):
        assert hfu.pick_mmproj(["mmproj-F32.gguf", "mmproj-BF16.gguf"], None) == ["mmproj-BF16.gguf"]
        assert hfu.pick_mmproj(["mmproj-F32.gguf"], None) == ["mmproj-F32.gguf"]

    def test_auto_with_unrecognised_precision_takes_first_sorted(self):
        assert hfu.pick_mmproj(["z-mmproj-Q8.gguf", "a-mmproj-Q8.gguf"], None) == ["a-mmproj-Q8.gguf"]

    def test_explicit_precision_is_case_insensitive(self):
        assert hfu.pick_mmproj(MMPROJ, "bf16") == ["mmproj-BF16.gguf"]

    def test_all_and_none(self):
        assert hfu.pick_mmproj(MMPROJ, "all") == MMPROJ
        assert hfu.pick_mmproj(MMPROJ, "none") == []

    def test_explicit_precision_not_present_raises(self):
        with pytest.raises(ValueError, match="F64"):
            hfu.pick_mmproj(MMPROJ, "F64")

    def test_no_mmproj_files_gives_empty_list_regardless_of_mode(self):
        assert hfu.pick_mmproj([], None) == []
        assert hfu.pick_mmproj([], "all") == []


class TestSelectQuants:
    quants = hfu.group_quants(UNSLOTH_FILES)

    def test_exact_name_case_insensitive(self):
        [q] = hfu.select_quants(self.quants, ["ud-q4_k_s"])
        assert q.name == "UD-Q4_K_S"

    def test_matches_file_stem_suffix_when_prefix_trimming_ate_a_token(self):
        # Repo where every quant is UD-*, so the trimmed name lacks 'UD-'
        files = [RepoFile("M-UD-Q4_K_S.gguf", 1), RepoFile("M-UD-Q5_K_S.gguf", 2)]
        quants = hfu.group_quants(files)
        [q] = hfu.select_quants(quants, ["UD-Q4_K_S"])
        assert q.files == ["M-UD-Q4_K_S.gguf"]

    def test_does_not_substring_match(self):
        with pytest.raises(ValueError, match="Q4_K"):
            hfu.select_quants(self.quants, ["Q4_K"])

    def test_error_lists_available_names(self):
        with pytest.raises(ValueError, match="UD-Q4_K_XL"):
            hfu.select_quants(self.quants, ["nope"])

    def test_multiple_specs_preserve_request_order(self):
        picked = hfu.select_quants(self.quants, ["Q8_0", "MXFP4_MOE"])
        assert [q.name for q in picked] == ["Q8_0", "MXFP4_MOE"]


class TestParseChoice:
    def test_single_number(self):
        assert hfu.parse_choice("3", 5) == [2]

    def test_comma_and_space_separated_dedup_in_order(self):
        assert hfu.parse_choice("3, 1,3 2", 5) == [2, 0, 1]

    def test_quit_returns_none(self):
        assert hfu.parse_choice("q", 5) is None
        assert hfu.parse_choice("", 5) is None

    @pytest.mark.parametrize("bad", ["0", "6", "abc", "1,x"])
    def test_out_of_range_or_garbage_raises(self, bad):
        with pytest.raises(ValueError):
            hfu.parse_choice(bad, 5)


class TestBuildCommand:
    def test_plain_model_download(self):
        assert hfu.build_command("org/m", "model", [], []) == ["hf", "download", "org/m"]

    def test_dataset_adds_repo_type(self):
        cmd = hfu.build_command("org/d", "dataset", [], [])
        assert cmd == ["hf", "download", "org/d", "--repo-type", "dataset"]

    def test_filenames_are_positional_after_repo(self):
        cmd = hfu.build_command("org/m", "model", ["a.gguf", "mmproj-F16.gguf"], [])
        assert cmd == ["hf", "download", "org/m", "a.gguf", "mmproj-F16.gguf"]

    def test_passthrough_args_appended_last(self):
        cmd = hfu.build_command("org/m", "model", ["a.gguf"], ["--revision", "v2", "--dry-run"])
        assert cmd == ["hf", "download", "org/m", "a.gguf", "--revision", "v2", "--dry-run"]


class TestParseArgs:
    def test_defaults(self):
        ns, extra = hfu.parse_args(["org/m"])
        assert ns.target == "org/m"
        assert ns.quant == [] and ns.mmproj is None and extra == []

    def test_quant_flag_accepts_repeats_and_commas(self):
        ns, _ = hfu.parse_args(["org/m", "-q", "Q8_0", "--quant", "Q4_K_M,Q4_K_S"])
        assert ns.quant == ["Q8_0", "Q4_K_M", "Q4_K_S"]

    def test_mmproj_flag(self):
        ns, _ = hfu.parse_args(["org/m", "--mmproj", "all"])
        assert ns.mmproj == "all"

    def test_unknown_options_and_their_values_pass_through_in_order(self):
        ns, extra = hfu.parse_args(["org/m", "--revision", "v2", "-q", "Q8_0", "--local-dir", "./x", "--dry-run"])
        assert ns.quant == ["Q8_0"]
        assert extra == ["--revision", "v2", "--local-dir", "./x", "--dry-run"]


class TestFormatSize:
    @pytest.mark.parametrize("n,expected", [
        (None, "?"), (0, "0 B"), (900, "900 B"), (1536, "1.5 KB"),
        (20 * GB, "20.0 GB"), (int(1.25 * 1024 ** 4), "1.25 TB"),
    ])
    def test_human_readable(self, n, expected):
        assert hfu.format_size(n) == expected


class TestPlanFilenames:
    quants = {q.name: q for q in hfu.group_quants(UNSLOTH_FILES)}

    def test_quant_files_then_mmproj_then_readme(self):
        names = hfu.plan_filenames(UNSLOTH_FILES, [self.quants["UD-Q4_K_S"]], mmproj_mode=None)
        assert names == ["Qwen3.6-35B-A3B-UD-Q4_K_S.gguf", "mmproj-F16.gguf", "README.md"]

    def test_sharded_quant_lists_every_shard(self):
        names = hfu.plan_filenames(UNSLOTH_FILES, [self.quants["BF16"]], mmproj_mode="none")
        assert names == [
            "BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf",
            "BF16/Qwen3.6-35B-A3B-BF16-00002-of-00002.gguf",
            "README.md",
        ]

    def test_text_only_repo_has_no_mmproj_even_with_all(self):
        files = [RepoFile("m-Q4_K_M.gguf", 1), RepoFile("m-Q8_0.gguf", 2)]
        [q4, _] = hfu.group_quants(files)
        assert hfu.plan_filenames(files, [q4], mmproj_mode="all") == ["m-Q4_K_M.gguf"]

    def test_readme_only_when_present_case_insensitive(self):
        files = [RepoFile("m-Q4_K_M.gguf", 1), RepoFile("readme.md", 1)]
        [q4] = hfu.group_quants(files)
        assert hfu.plan_filenames(files, [q4], mmproj_mode=None) == ["m-Q4_K_M.gguf", "readme.md"]


class TestParseHfInput:
    @pytest.mark.parametrize("inp,expected", [
        ("org/name", ("org/name", "model")),
        ("models/org/name", ("org/name", "model")),
        ("datasets/org/name", ("org/name", "dataset")),
        ("spaces/org/name", ("org/name", "space")),
        ("https://huggingface.co/org/name", ("org/name", "model")),
        ("https://huggingface.co/datasets/org/name", ("org/name", "dataset")),
        ("https://huggingface.co/spaces/org/name", ("org/name", "space")),
        ("http://huggingface.co/datasets/org/name", ("org/name", "dataset")),
        ("https://huggingface.co/org/name/tree/main", ("org/name", "model")),
        ("https://huggingface.co/org/name/blob/main/README.md", ("org/name", "model")),
        ("https://huggingface.co/datasets/org/name/tree/main/data", ("org/name", "dataset")),
        ("https://huggingface.co/org/name?library=gguf", ("org/name", "model")),
        ("https://hf.co/org/name", ("org/name", "model")),
        ("  org/name  ", ("org/name", "model")),
    ])
    def test_accepted_forms(self, inp, expected):
        assert hfu.parse_hf_input(inp) == expected

    @pytest.mark.parametrize("bad", ["name", "", "org/", "https://example.com/org/name"])
    def test_rejected_forms(self, bad):
        with pytest.raises(ValueError):
            hfu.parse_hf_input(bad)


class TestTableRows:
    def test_rows_have_index_name_size_and_shard_count(self):
        quants = hfu.group_quants(UNSLOTH_FILES)
        rows = hfu.table_rows(quants)
        assert rows[0][0] == "1"
        bf16 = next(r for r in rows if r[1] == "BF16")
        assert bf16[2] == "70.0 GB"
        assert bf16[3] == "2 files"
        single = next(r for r in rows if r[1] == "Q8_0")
        assert single[3] == ""


class TestCacheStatus:
    """probe_cache compares the Hub's content hash with what the local hub cache holds."""

    REPO = "org/Model-GGUF"

    @staticmethod
    def _sha(content: bytes) -> str:
        import hashlib
        return hashlib.sha256(content).hexdigest()

    def _hub(self, tmp_path, files):
        from tests.hub_fixture import add_repo
        add_repo(tmp_path, self.REPO, files)
        return tmp_path

    def test_current_when_hub_blob_is_cached(self, tmp_path):
        hub = self._hub(tmp_path, {"m-Q4_K_M.gguf": b"q4"})
        files = [RepoFile("m-Q4_K_M.gguf", 2, self._sha(b"q4"))]
        assert hfu.probe_cache(hub, self.REPO, files) == {"m-Q4_K_M.gguf": "current"}

    def test_stale_when_upstream_content_changed(self, tmp_path):
        hub = self._hub(tmp_path, {"m-Q4_K_M.gguf": b"old"})
        files = [RepoFile("m-Q4_K_M.gguf", 3, self._sha(b"new"))]
        assert hfu.probe_cache(hub, self.REPO, files) == {"m-Q4_K_M.gguf": "stale"}

    def test_absent_files_are_omitted(self, tmp_path):
        hub = self._hub(tmp_path, {"m-Q4_K_M.gguf": b"q4"})
        files = [RepoFile("m-Q8_0.gguf", 2, self._sha(b"q8"))]
        assert hfu.probe_cache(hub, self.REPO, files) == {}

    def test_unknown_hash_reports_cached(self, tmp_path):
        hub = self._hub(tmp_path, {"m-Q4_K_M.gguf": b"q4"})
        files = [RepoFile("m-Q4_K_M.gguf", 2)]
        assert hfu.probe_cache(hub, self.REPO, files) == {"m-Q4_K_M.gguf": "cached"}

    def test_plain_copy_without_symlinks_reports_cached(self, tmp_path):
        from tests.hub_fixture import folder
        snap = tmp_path / folder(self.REPO) / "snapshots" / "abc"
        snap.mkdir(parents=True)
        (snap / "m-Q4_K_M.gguf").write_bytes(b"q4")
        files = [RepoFile("m-Q4_K_M.gguf", 2, self._sha(b"q4"))]
        assert hfu.probe_cache(tmp_path, self.REPO, files) == {"m-Q4_K_M.gguf": "cached"}

    def test_repo_not_in_cache(self, tmp_path):
        files = [RepoFile("m-Q4_K_M.gguf", 2, self._sha(b"q4"))]
        assert hfu.probe_cache(tmp_path, self.REPO, files) == {}


class TestCacheNote:
    def _quant(self, n):
        return hfu.Quant("Q4", [f"m-Q4-0000{i}-of-0000{n}.gguf" for i in range(1, n + 1)], n)

    def test_all_current(self):
        q = self._quant(1)
        assert hfu.cache_note(q, {q.files[0]: "current"}) == "✓ cached"

    def test_any_stale_means_update_available(self):
        q = self._quant(2)
        assert hfu.cache_note(q, {q.files[0]: "current", q.files[1]: "stale"}) == "↻ update available"

    def test_partial_shards(self):
        q = self._quant(3)
        assert hfu.cache_note(q, {q.files[0]: "current"}) == "partial (1/3)"

    def test_not_cached(self):
        assert hfu.cache_note(self._quant(1), {}) == ""

    def test_table_rows_combine_shard_and_cache_notes(self):
        quants = hfu.group_quants(UNSLOTH_FILES)
        bf16 = next(q for q in quants if q.name == "BF16")
        q8 = next(q for q in quants if q.name == "Q8_0")
        states = {f: "current" for f in bf16.files} | {q8.files[0]: "stale"}
        rows = {r[1]: r for r in hfu.table_rows(quants, states)}
        assert rows["BF16"][3] == "2 files · ✓ cached"
        assert rows["Q8_0"][3] == "↻ update available"
        assert rows["MXFP4_MOE"][3] == ""
