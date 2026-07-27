# Copyright 2026 Jiang Yongchun
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_public_files_exist() -> None:
    for relative in (
        "README.md",
        "LICENSE",
        "NOTICE",
        "CITATION.cff",
        "CMakeLists.txt",
        "patches/cuml-26.06.00/coherent-fp16.patch",
        "src/portable_fp16_rbf_backend.cu",
        "artifact/configs/smoke-synthetic.toml",
    ):
        assert (ROOT / relative).is_file(), relative


def test_tracked_text_has_no_private_workspace_tokens() -> None:
    user = "jiang" + "yongchun"
    forbidden = (
        "/home/data/" + user,
        "/Users/" + user,
        "pro6000" + "-8-root",
        "fanshen" + "king",
        "test" + "123",
    )
    suffixes = {
        ".md",
        ".py",
        ".toml",
        ".json",
        ".sh",
        ".cu",
        ".cuh",
        ".patch",
        ".txt",
    }
    for path in ROOT.rglob("*"):
        if any(part == "build" or part.startswith("build-") for part in path.parts):
            continue
        if any(part in {".git", ".venv", "results"} for part in path.parts):
            continue
        public_text = path.suffix in suffixes or path.name in {"NOTICE", "CMakeLists.txt"}
        if path.is_file() and public_text:
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert not any(token in text for token in forbidden), path
