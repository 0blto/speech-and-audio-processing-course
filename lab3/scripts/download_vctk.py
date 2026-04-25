import argparse
import io
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

VCTK_ZIP_URL = (
    "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _stream_download(url: str, dest_zip: Path, chunk: int = 1 << 20) -> None:
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=dest_zip.name)
        with open(dest_zip, "wb") as f:
            for c in r.iter_content(chunk_size=chunk):
                if c:
                    f.write(c)
                    pbar.update(len(c))
        pbar.close()


def _unzip(extract_to: Path, dest_zip: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest_zip, "r") as zf:
        zf.extractall(extract_to)


def _find_vctk_root(base: Path) -> Path | None:
    for pat in (
        "VCTK-Corpus-0.92",
        "VCTK",
        "VCTK_Corpus_0.92",
    ):
        p = base / pat
        if p.is_dir() and (p / "txt").is_dir():
            return p
    for txt in base.rglob("txt"):
        if txt.is_dir() and (txt.parent / "wav48_silence_trimmed").is_dir():
            return txt.parent
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Скачать VCTK (≈10 GiB) и распаковать у txt/ + wav48_silence_trimmed/."
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Куда скачать zip и куда извлечь",
    )
    parser.add_argument(
        "--url",
        default=VCTK_ZIP_URL,
        help="URL zip (зеркало, если ed.ac.uk недоступен)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Только разархивировать, если VCTK-Corpus-0.92.zip уже лежит в out-root",
    )
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    zpath = out_root / "VCTK-Corpus-0.92.zip"

    vpre = _find_vctk_root(out_root)
    if vpre is not None:
        (out_root / "VCTK_root.txt").write_text(str(vpre) + "\n", encoding="utf-8")
        print(
            f"VCTK уже на месте: {vpre}\n"
            f"Далее: python scripts/prepare_vctk_speaker.py --vctk-root {vpre}"
        )
        return

    if not args.no_download:
        print("Скачивание (долго, ≈10 GiB)…", file=sys.stderr)
        _stream_download(args.url, zpath)
    else:
        if not zpath.is_file():
            raise SystemExit(f"Нет {zpath}. Уберите --no-download или положите zip вручную.")

    vroot = _find_vctk_root(out_root)
    if vroot is None and zpath.is_file():
        print("Распаковка (может занять несколько минут)…", file=sys.stderr)
        _unzip(out_root, zpath)
        vroot = _find_vctk_root(out_root)
    if vroot is None:
        raise SystemExit(
            "Не найдена структура VCTK. Ожидается распаковка с каталогами txt/ и "
            "wav48_silence_trimmed/ — проверьте zip и путь out-root."
        )
    (out_root / "VCTK_root.txt").write_text(str(vroot) + "\n", encoding="utf-8")
    print(f"OK. Корень VCTK: {vroot}\n"
          f"Далее: python scripts/prepare_vctk_speaker.py --vctk-root {vroot}")


if __name__ == "__main__":
    main()
