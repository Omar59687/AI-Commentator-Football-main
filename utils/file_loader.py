from pathlib import Path


def load_text(path: str) -> str:

    file = Path(path)

    if not file.exists():
        raise FileNotFoundError(f"{path} not found.")

    return file.read_text(encoding="utf-8")