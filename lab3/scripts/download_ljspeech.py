import argparse
import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

DEFAULT_URL = "https://data.keith.it/LJSpeech-1.1.tar.bz2"
EXPECTED_MIN_BYTES = 2_000_000_000

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=(30, 120)) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        chunk = 1024 * 1024
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="Скачивание"
        ) as pbar:
            for part in r.iter_content(chunk):
                if part:
                    f.write(part)
                    pbar.update(len(part))


def main() -> None:
    parser = argparse.ArgumentParser(description="Загрузка LJSpeech-1.1")
    parser.add_argument("--url", default=DEFAULT_URL, help="URL архива .tar.bz2")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Каталог для данных (по умолчанию ./data)",
    )
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()

    if (args.data_dir / "LJSpeech-1.1" / "metadata.csv").is_file():
        print(f"Уже есть: {args.data_dir / 'LJSpeech-1.1' / 'metadata.csv'} — пропуск загрузки.")
        return

    archive = args.data_dir / "LJSpeech-1.1.tar.bz2"
    if not archive.is_file():
        print(f"Загрузка из {args.url} …")
        try:
            download_file(args.url, archive)
        except Exception as e:
            print(
                "Не удалось скачать автоматически. Скачайте LJSpeech-1.1.tar.bz2 вручную и положите в",
                archive,
                f"\nОшибка: {e}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        print(f"Используется локальный архив: {archive}")

    if archive.stat().st_size < EXPECTED_MIN_BYTES:
        print(
            "Предупреждение: размер архива меньше ожидаемого — возможно, загрузка оборвалась.",
            file=sys.stderr,
        )

    print("Распаковка…")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:bz2") as tf:
        tf.extractall(path=args.data_dir)

    meta = args.data_dir / "LJSpeech-1.1" / "metadata.csv"
    if not meta.is_file():
        print(f"Не найден {meta} после распаковки.", file=sys.stderr)
        sys.exit(1)
    print(f"Готово: {meta.parent}")


if __name__ == "__main__":
    main()
