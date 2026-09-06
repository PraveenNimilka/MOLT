from pathlib import Path


def test_compiler_cache_uses_short_existing_directories(tmp_path, monkeypatch):
    from molt_stream.kernels.compiler import configure_windows_compiler_cache
    monkeypatch.chdir(tmp_path)
    for name in ("TRITON_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)
    root = configure_windows_compiler_cache()
    assert root == (tmp_path / ".c").resolve()
    assert Path(__import__("os").environ["TRITON_CACHE_DIR"]) == root / "t"
    assert Path(__import__("os").environ["TORCHINDUCTOR_CACHE_DIR"]) == root / "i"
    assert (root / "t").is_dir() and (root / "i").is_dir()
