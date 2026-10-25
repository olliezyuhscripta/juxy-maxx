import pathlib

from huggingface_hub import HfApi

WORK = pathlib.Path("/marimo/juxy30b")
REPO = "zxcxzczxcxzc/juxy-maxx-30b"
TOKEN = pathlib.Path("/marimo/.secrets/hf_token").read_text(encoding="utf-8").strip()


def main():
    api = HfApi(token=TOKEN)
    url = api.create_repo(REPO, repo_type="model", exist_ok=True, private=False)
    print("repo:", url)

    api.upload_folder(
        folder_path=str(WORK / "adapter"),
        repo_id=REPO,
        repo_type="model",
        token=TOKEN,
        path_in_repo=".",
        ignore_patterns=["*.pt", "optimizer*", "*.pth"],
    )
    for name in ["README.md", "eval.json", "train.log"]:
        path = WORK / name
        if path.exists():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"training/{name}" if name != "README.md" else name,
                repo_id=REPO,
                repo_type="model",
                token=TOKEN,
            )
            print("uploaded", name)

    info = api.model_info(REPO)
    print("id:", info.id)
    print("files:", [s.rfilename for s in info.siblings])


if __name__ == "__main__":
    main()
