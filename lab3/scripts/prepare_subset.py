import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LJ = PROJECT_ROOT / "data" / "LJSpeech-1.1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ljspeech-root",
        type=Path,
        default=DEFAULT_LJ,
        help="Корень LJSpeech-1.1 (где лежат wavs/ и metadata.csv)",
    )
    parser.add_argument("--num-samples", type=int, default=500, help="Сколько строк взять")
    args = parser.parse_args()
    root = args.ljspeech_root.resolve()
    meta = root / "metadata.csv"
    if not meta.is_file():
        raise SystemExit(f"Нет файла {meta}. Сначала запустите scripts/download_ljspeech.py")

    lines: list[str] = []
    with open(meta, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.num_samples:
                break
            lines.append(line.rstrip("\n"))

    out = root / "metadata_subset.csv"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Записано {len(lines)} строк в {out}")


if __name__ == "__main__":
    main()
