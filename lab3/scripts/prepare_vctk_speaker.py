import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SPEAKER = "p360"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--vctk-root",
        type=Path,
        default=None,
        help="Корень VCTK (с txt/ и wav48_silence_trimmed/). Читается data/VCTK_root.txt, если задано.",
    )
    ap.add_argument(
        "--speaker",
        default=DEFAULT_SPEAKER,
        help="ID диктора (каталог в txt/ и в wavs), напр. p360",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Куда копировать (по умолчанию data/VCTK_{speaker})",
    )
    ap.add_argument(
        "--link",
        action="store_true",
        help="Вместо копии использовать symlink (только на Linux/macOS, на Windows копия)",
    )
    args = ap.parse_args()

    vctk = args.vctk_root
    if vctk is None:
        hint = PROJECT_ROOT / "data" / "VCTK_root.txt"
        if hint.is_file():
            vctk = Path(hint.read_text(encoding="utf-8").strip())
        else:
            raise SystemExit("Укажите --vctk-root или сначала: python scripts/download_vctk.py")
    vctk = vctk.resolve()
    sp = args.speaker
    out = (args.out or (PROJECT_ROOT / "data" / f"VCTK_{sp}")).resolve()

    src_txt = vctk / "txt" / sp
    src_wav = vctk / "wav48_silence_trimmed" / sp
    if not src_txt.is_dir():
        raise SystemExit(f"Нет {src_txt}")
    if not src_wav.is_dir():
        raise SystemExit(f"Нет {src_wav}")

    out_txt = out / "txt" / sp
    out_wav = out / "wav48_silence_trimmed" / sp
    for p in (out_txt.parent, out_wav.parent):
        p.mkdir(parents=True, exist_ok=True)
    if out_txt.is_dir() or out_wav.is_dir():
        shutil.rmtree(out_txt, ignore_errors=True)
        shutil.rmtree(out_wav, ignore_errors=True)

    if args.link and sys.platform != "win32":
        out_txt.symlink_to(src_txt, target_is_directory=True)
        out_wav.symlink_to(src_wav, target_is_directory=True)
    else:
        if args.link and sys.platform == "win32":
            print("Symlink на Windows пропущен, выполняю копирование.", file=sys.stderr)
        shutil.copytree(src_txt, out_txt)
        shutil.copytree(src_wav, out_wav)

    n = len(list(out_txt.glob("*.txt")))
    print(
        f"OK: {n} сегментов диктора {sp} -> {out}\n"
        f"Обучение: python scripts/finetune.py --exp A --vctk-single-root {out}"
    )


if __name__ == "__main__":
    main()
