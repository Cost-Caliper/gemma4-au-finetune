"""Upload model artifacts from the au-gemma4 Modal volume to Hugging Face.

  modal run modal_hf_upload.py --repo dennisonb/xxx --src /vol/tuned-v5-q8.gguf [--private]
  modal run modal_hf_upload.py --repo dennisonb/xxx --src /vol/adapter-v5 --readme-path /vol/cards/v5.md

HF token passed via the 'hf-write' Modal secret (HF_TOKEN).
"""
import modal

app = modal.App("au-hf-upload")
vol = modal.Volume.from_name("au-gemma4")
img = (modal.Image.debian_slim(python_version="3.12")
       .pip_install("huggingface_hub[hf_transfer]")
       .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}))


@app.function(image=img, volumes={"/vol": vol}, timeout=7200,
              secrets=[modal.Secret.from_name("hf-write")])
def upload(repo: str, src: str, private: bool = True, readme_text: str = ""):
    import os
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, private=private, exist_ok=True, repo_type="model")
    if readme_text:
        api.upload_file(path_or_fileobj=readme_text.encode(), path_in_repo="README.md",
                        repo_id=repo)
    if os.path.isdir(src):
        api.upload_folder(folder_path=src, repo_id=repo,
                          ignore_patterns=["*.lock", ".cache*"])
    else:
        api.upload_file(path_or_fileobj=src, path_in_repo=os.path.basename(src), repo_id=repo)
    print(f"UPLOADED {src} -> {repo}")
    return repo
