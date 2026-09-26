from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "https://afproject.org/media/genreg/crm/dataset/crm.zip"
EXPECTED_MD5 = "d576c477b92343fed45fbc6bce2b5cac"
REFERENCE_URL = (
    "https://raw.githubusercontent.com/afproject-org/afproject/"
    "master/datasets/genreg/crm/ids.json"
)

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
ARCHIVE = DATASETS_DIR / "crm.zip"
DATASET_DIR = DATASETS_DIR / "crm"
REFERENCE_FILE = DATASET_DIR / "ids.json"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def download_dataset() -> None:
    if DATASET_DIR.is_dir() and any(DATASET_DIR.rglob("*.fasta")):
        print(f"CRM dataset already exists: {DATASET_DIR}")
        return

    download_file(DATASET_URL, ARCHIVE)

    actual = md5(ARCHIVE)
    if actual != EXPECTED_MD5:
        raise RuntimeError(
            f"CRM archive MD5 mismatch: expected {EXPECTED_MD5}, got {actual}"
        )

    print(f"Extracting {ARCHIVE} -> {DATASETS_DIR}")
    with zipfile.ZipFile(ARCHIVE) as archive:
        archive.extractall(DATASETS_DIR)

    if not DATASET_DIR.is_dir():
        raise RuntimeError(
            f"Expected extracted dataset at {DATASET_DIR}, but it was not found."
        )

    print(f"Saved CRM dataset to: {DATASET_DIR}")


def download_reference() -> None:
    if REFERENCE_FILE.is_file():
        print(f"CRM reference already exists: {REFERENCE_FILE}")
        return

    download_file(REFERENCE_URL, REFERENCE_FILE)
    print(f"Saved CRM reference to: {REFERENCE_FILE}")


def main() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    download_dataset()
    download_reference()


if __name__ == "__main__":
    main()
