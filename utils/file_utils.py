from pathlib import Path


def save_text(text: str, output_path: str):

    path = Path(output_path)

    # Create the folder if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write(text)

    print(f"Saved: {path}")