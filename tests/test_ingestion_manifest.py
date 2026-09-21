"""The download manifest must skip re-downloading unchanged files and fail loudly, never
silently, when a download errors."""

import pytest
import requests

from pdm.ingestion.manifest import download_file

pytestmark = pytest.mark.fast


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        pass


def test_download_file_is_idempotent(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_get(url, timeout=120):
        calls["n"] += 1
        return _FakeResponse(b"hello world")

    monkeypatch.setattr("pdm.ingestion.manifest.requests.get", fake_get)

    dest = tmp_path / "file.bin"
    manifest_path = tmp_path / "manifest.json"

    download_file(
        "http://example.com/file",
        dest,
        source_page="http://example.com",
        manifest_path=manifest_path,
    )
    assert calls["n"] == 1
    assert dest.read_bytes() == b"hello world"

    download_file(
        "http://example.com/file",
        dest,
        source_page="http://example.com",
        manifest_path=manifest_path,
    )
    assert calls["n"] == 1, "second call should be served from the manifest cache, no network call"


def test_download_file_redownloads_if_local_copy_was_tampered_with(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_get(url, timeout=120):
        calls["n"] += 1
        return _FakeResponse(b"hello world")

    monkeypatch.setattr("pdm.ingestion.manifest.requests.get", fake_get)

    dest = tmp_path / "file.bin"
    manifest_path = tmp_path / "manifest.json"
    download_file(
        "http://example.com/file",
        dest,
        source_page="http://example.com",
        manifest_path=manifest_path,
    )
    dest.write_bytes(b"corrupted")

    download_file(
        "http://example.com/file",
        dest,
        source_page="http://example.com",
        manifest_path=manifest_path,
    )
    assert calls["n"] == 2
    assert dest.read_bytes() == b"hello world"


def test_download_file_fails_loudly_and_names_source_page(tmp_path, monkeypatch):
    def fake_get(url, timeout=120):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr("pdm.ingestion.manifest.requests.get", fake_get)

    dest = tmp_path / "file.bin"
    manifest_path = tmp_path / "manifest.json"

    with pytest.raises(RuntimeError, match="https://example.com/source-page"):
        download_file(
            "http://example.com/file",
            dest,
            source_page="https://example.com/source-page",
            manifest_path=manifest_path,
        )
    assert not dest.exists()
