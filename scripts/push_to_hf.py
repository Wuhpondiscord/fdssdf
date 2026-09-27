from __future__ import annotations

import os
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = os.getenv("HF_SPACE_REPO", "wuhp/ghtest").strip()
TOKEN = os.getenv("HF_TOKEN")
if not TOKEN:
    raise SystemExit("HF_TOKEN is missing. Add it as a GitHub Actions secret.")
api = HfApi(token=TOKEN)
api.create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="gradio", exist_ok=True)
commit = api.upload_folder(repo_id=REPO_ID, repo_type="space", folder_path=str(ROOT), commit_message="Mirror Wuhpondiscord/fdssdf main to Hugging Face", ignore_patterns=[".git/**", ".github/**", ".venv/**", "__pycache__/**", ".pytest_cache/**", "tests/**"])
print(f"Mirrored to {REPO_ID}: {commit.oid}")
