"""SHA-256 download manifest shared by every dataset loader, so `make data` is idempotent.

Network calls only happen inside :func:`download_file`, never at import time. A failed download
raises rather than falling back to any kind of synthetic data — a silent substitution would be
far worse than a loud crash naming where to get the file by hand.
"""

import hashlib
import json
from pathlib import Path

import requests


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def download_file(
    url: str,
    dest: Path,
    *,
    source_page: str,
    manifest_path: Path,
    timeout: int = 120,
    session: requests.Session | None = None,
) -> Path:
    """Download ``url`` to ``dest`` unless the manifest shows it is already present and unchanged.

    Raises :class:`RuntimeError` on any HTTP failure, naming ``source_page`` so a human can find a
    mirror or download manually. Never substitutes synthetic data for a failed download.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    key = dest.as_posix()

    cached = manifest.get(key)
    if dest.exists() and cached is not None and cached.get("sha256") == _sha256(dest):
        return dest

    http = session or requests
    try:
        response = http.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to download {url!r}: {exc}. This dataset is required — no synthetic "
            f"fallback is generated. See the source page for a mirror or manual download: "
            f"{source_page}"
        ) from exc

    dest.write_bytes(response.content)
    manifest[key] = {
        "url": url,
        "sha256": _sha256(dest),
        "bytes": len(response.content),
    }
    save_manifest(manifest_path, manifest)
    return dest


def record_manifest_entry(dest: Path, *, source: str, manifest_path: Path) -> None:
    """Record a manifest entry for a file produced by a non-URL source (e.g. an API client)."""
    manifest = load_manifest(manifest_path)
    manifest[dest.as_posix()] = {
        "url": source,
        "sha256": _sha256(dest),
        "bytes": dest.stat().st_size,
    }
    save_manifest(manifest_path, manifest)


def is_cached(dest: Path, *, manifest_path: Path) -> bool:
    manifest = load_manifest(manifest_path)
    cached = manifest.get(dest.as_posix())
    return bool(dest.exists() and cached is not None and cached.get("sha256") == _sha256(dest))
