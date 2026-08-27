import os
import sys

def deploy():
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub")
        from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN")
    if not token and len(sys.argv) > 1:
        token = sys.argv[1]

    if not token:
        print("❌ Error: No Hugging Face API Token provided.")
        print("Usage: python3 deploy_to_hf.py <YOUR_HF_WRITE_TOKEN>")
        sys.exit(1)

    api = HfApi(token=token)
    
    try:
        user_info = api.whoami()
        username = user_info.get("name") or user_info.get("user")
        print(f"👤 Logged in to Hugging Face as: {username}")
    except Exception as e:
        print(f"❌ Failed to verify token: {e}")
        sys.exit(1)

    # 1. Upload Model Hub Repositories
    mbert_repo = f"{username}/drishtinet-mbert"
    muril_repo = f"{username}/drishtinet-muril"
    space_repo = f"{username}/drishtinet-ai-static"

    print(f"📦 Uploading fine-tuned mBERT weights to Model Hub ({mbert_repo})...")
    create_repo(mbert_repo, repo_type="model", exist_ok=True, token=token)
    if os.path.exists("./mbert"):
        api.upload_folder(
            folder_path="./mbert",
            repo_id=mbert_repo,
            repo_type="model",
            token=token
        )
        print(f"✅ mBERT model live at: https://huggingface.co/{mbert_repo}")

    print(f"📦 Uploading fine-tuned MuRIL weights to Model Hub ({muril_repo})...")
    create_repo(muril_repo, repo_type="model", exist_ok=True, token=token)
    if os.path.exists("./muril"):
        api.upload_folder(
            folder_path="./muril",
            repo_id=muril_repo,
            repo_type="model",
            token=token
        )
        print(f"✅ MuRIL model live at: https://huggingface.co/{muril_repo}")

    # 2. Deploy Web App Space
    print(f"🚀 Deploying DrishtiNet AI Space to: https://huggingface.co/spaces/{space_repo}...")
    create_repo(space_repo, repo_type="space", space_sdk="static", exist_ok=True, token=token)

    # Clean legacy static files if any
    for file_to_del in ["index.html", "style.css"]:
        try:
            api.delete_file(path_in_repo=file_to_del, repo_id=space_repo, repo_type="space", token=token)
        except Exception:
            pass

    api.upload_folder(
        folder_path=".",
        repo_id=space_repo,
        repo_type="space",
        token=token,
        ignore_patterns=[".git*", "*.pyc", "__pycache__", ".DS_Store", "deploy_to_hf.py", ".agents", "mbert/*", "muril/*"]
    )

    print("\n🎉 Deployment completed successfully!")
    print(f"🌐 Access your live DrishtiNet AI Space here: https://huggingface.co/spaces/{space_repo}")

if __name__ == "__main__":
    deploy()
