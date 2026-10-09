from pathlib import Path
from urllib.parse import quote


def image_file_url(image_path: str) -> str:
    filename = Path(image_path.replace("\\", "/")).name
    if not filename:
        raise ValueError("Image path does not contain a usable filename")
    return f"/images/files/{quote(filename, safe='')}"
