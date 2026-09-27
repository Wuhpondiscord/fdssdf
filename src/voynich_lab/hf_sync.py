from __future__ import annotations

import os
import re
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_SPACE_REPO = "wuhp/ghtest"
DEFAULT_SPACE_URL = "https://huggingface.co/spaces/wuhp/ghtest/tree/main"


def normalize_space_target(value: str | None) -> str:
    raw = (value or "").strip()
    if not raw:
        return DEFAULT_SPACE_REPO
    match = re.search(r"huggingface\.co/spaces/([^/]+/[^/]+)", raw)
    if match:
        return match.group(1)
    return raw.rstrip("/")


def configured_space_repo() -> str:
    return normalize_space_target(os.getenv("HF_SPACE_REPO", DEFAULT_SPACE_REPO))


def configured_space_value() -> str:
    return os.getenv("HF_SPACE_URL", DEFAULT_SPACE_URL).strip() or DEFAULT_SPACE_URL


def sync_status() -> str:
    return (
        f"Target URL: {configured_space_value()}\n"
        f"Resolved repo: {configured_space_repo()}\n"
        f"HF_TOKEN: {'configured' if os.getenv('HF_TOKEN') else 'missing'}\n"
        "Primary sync path: GitHub Actions mirror on push to main."
    )


def push_snapshot(target: str | None = None) -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not configured. Add it as a Hugging Face Space secret for in-app pushes, "
            "and as a GitHub Actions secret for automatic mirroring."
        )
    repo_id = normalize_space_target(target or configured_space_repo())
    root = Path(__file__).resolve().parents[2]
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(root),
        commit_message="Sync Voynich Structure Lab snapshot",
        ignore_patterns=[
            ".git/**",
            ".github/**",
            ".venv/**",
            "__pycache__/**",
            ".pytest_cache/**",
            "tests/**",
        ],
    )
    return f"Pushed snapshot to {repo_id}\nCommit: {commit.oid}"
