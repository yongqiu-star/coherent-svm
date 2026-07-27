# Copyright 2026 Jiang Yongchun
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repository: Path) -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repository), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"revision": "unavailable", "dirty": None}


def public_manifest(
    *,
    repository: Path,
    config_id: str,
    config_digest: str,
    library: Path,
    dataset_id: str,
) -> dict[str, Any]:
    """Return provenance without hostnames, UUIDs, private remotes, or local paths."""
    return {
        "schema": "coherent-svm-run-manifest-v1",
        "config_id": config_id,
        "config_sha256": config_digest,
        "source": git_revision(repository),
        "library_sha256": sha256_file(library),
        "dataset_id": dataset_id,
        "python": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

