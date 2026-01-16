
import os
from huggingface_hub import HfApi, create_repo

# Config
HF_TOKEN = os.environ.get("HF_TOKEN")
USERNAME = "ananthgv-usk"
REPO_NAME = "orpheus-3b-ft-sph45"
REPO_ID = f"{USERNAME}/{REPO_NAME}"
LOCAL_DIR = "/workspace/orpheus_fft_final"

print(f"Preparing to upload {LOCAL_DIR} to {REPO_ID}...")

if not HF_TOKEN:
    print("Error: HF_TOKEN not found!")
    exit(1)

api = HfApi(token=HF_TOKEN)

# Create Repo
try:
    print(f"Creating repo {REPO_ID} (if not exists)...")
    create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, token=HF_TOKEN)
except Exception as e:
    print(f"Repo creation warning (might exist): {e}")

# Upload
print("Starting upload (this may take a while)...")
try:
    api.upload_folder(
        folder_path=LOCAL_DIR,
        repo_id=REPO_ID,
        repo_type="model",
        ignore_patterns=["checkpoint-*"] # Upload final model only? Or checkpoints too?
        # User asked for "full finetuned model". Usually confirms checkpoints.
        # But checkpoints are huge.
        # "save_total_limit=1" means only 1 checkpoint exists.
        # Uploading everything is safer.
    )
    print("✅ Upload Complete!")
except Exception as e:
    print(f"❌ Upload Failed: {e}")
    exit(1)
