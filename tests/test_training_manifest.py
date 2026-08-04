from quant.models.training import _git_commit


def test_build_commit_fallback_for_git_archive(tmp_path) -> None:
    commit = "0123456789abcdef0123456789abcdef01234567"
    (tmp_path / "BUILD_COMMIT").write_text(commit + "\n", encoding="utf-8")

    assert _git_commit(tmp_path) == commit
