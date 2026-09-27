from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_SPACE = "wuhp/ghtest"


def configured_space() -> str:
    return os.getenv("HF_SPACE_REPO", DEFAULT_SPACE).strip() or DEFAULT_SPACE


def sync_status() -> str:
    return f"Target: {configured_space()}\nHF_TOKEN: {'configured' if os.getenv('HF_TOKEN') else 'missing'}\nPrimary sync path: GitHub Actions mirror on push to main."


def push_snapshot(repo_id: str | None = None) -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not configured. Add it as a Hugging Face Space secret for in-app pushes, and as a GitHub Actions secret for automatic mirroring.")
    repo_id = (repo_id or configured_space()).strip()
    root = Path(__file__).resolve().parents[2]
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    commit = api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(root), commit_message="Sync Voynich Structure Lab snapshot", ignore_patterns=[".git/**", ".github/**", ".venv/**", "__pycache__/**", ".pytest_cache/**", "tests/**"])
    return f"Pushed snapshot to {repo_id}\nCommit: {commit.oid}"
