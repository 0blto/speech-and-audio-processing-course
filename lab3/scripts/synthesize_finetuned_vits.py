import argparse
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _espeak() -> bool:
    import shutil

    return bool(shutil.which("espeak-ng") or shutil.which("espeak"))


def _load_tts_for_run(
    run_dir: Path, checkpoint: Path | None, use_gpu: bool, progress_bar: bool, cache: Path
):
    from TTS.api import TTS
    from TTS.config import load_config

    cfg = run_dir / "config.json"
    if not cfg.is_file():
        raise SystemExit(f"В {run_dir} нет config.json. Укажите каталог --run-dir из runs/.")

    cands = [checkpoint] if checkpoint is not None else []
    cands += [
        run_dir / "best_model.pth",
        *sorted(run_dir.glob("**/*.pth"), key=lambda p: p.stat().st_mtime, reverse=True),
    ]
    cpath: Path | None = None
    for p in cands:
        if p is not None and p.is_file():
            cpath = p
            break
    if cpath is None:
        raise SystemExit(
            f"Нет .pth в {run_dir}. Дождитесь save_step / конца fit или укажите --checkpoint."
        )

    def _load(path_m: str, c_m: str | Path):
        return TTS(
            model_path=path_m,
            config_path=str(c_m) if c_m is not None else None,
            gpu=use_gpu,
            progress_bar=progress_bar,
        )

    mname = (load_config(str(cfg)) or type("C", (), {})).model
    if _espeak():
        return _load(str(cpath), cfg)

    print(
        "eSpeak в PATH нет: для совместимости с checkpoint попробуем снять use_phonemes "
        "как в inference_compare. Лучше установить espeak-ng.",
        file=sys.stderr,
    )
    config = load_config(str(cfg))
    if getattr(config, "use_phonemes", False):
        config.use_phonemes = False
        if hasattr(config, "phonemizer"):
            config.phonemizer = ""
        config.text_cleaner = "english_cleaners"
        safe = str(cpath).replace(":", "_").replace("\\", "_")
        sub = cache / f"infer_{Path(safe).name}.json"
        sub.parent.mkdir(parents=True, exist_ok=True)
        config.save_json(str(sub))
        return _load(str(cpath), sub)
    return _load(str(cpath), cfg)


def _chunks(text: str, max_c: int = 180) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) <= max_c:
        return [text] if text else []
    parts = re.split(r"(?<=[.!?])\s+", text)
    r: list[str] = []
    b = ""
    for p in parts:
        if not p:
            continue
        if len(b) + len(p) + 1 <= max_c:
            b = (b + " " + p).strip()
        else:
            if b:
                r.append(b)
            b = p
    if b:
        r.append(b)
    return r


def _synth_wav(tts, text: str) -> tuple[np.ndarray, int]:
    synth = tts.synthesizer
    if synth is None:
        raise RuntimeError("synthesizer is None")
    ch = _chunks(text)
    parts: list[np.ndarray] = []
    sr: int = 22050
    for c in ch:
        w = synth.tts(text=c, split_sentences=True)
        parts.append(np.asarray(w, dtype=np.float32))
        sr = int(getattr(synth, "output_sample_rate", sr))
    if not parts:
        return np.array([], dtype=np.float32), sr
    if len(parts) == 1:
        return parts[0], sr
    z = int(0.15 * sr)
    o: list[np.ndarray] = []
    for i, p in enumerate(parts):
        o.append(p)
        if i < len(parts) - 1:
            o.append(np.zeros(z, dtype=np.float32))
    return np.concatenate(o), sr


def _save_mel(wav: np.ndarray, sr: int, p: Path) -> None:
    import librosa
    import librosa.display

    y = wav.astype(np.float64)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel: {p.stem}")
    plt.tight_layout()
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=120)
    plt.close()


def _save_linear_stft(wav: np.ndarray, sr: int, p: Path, n_fft: int = 1024, hop: int = 256) -> None:
    import librosa
    import librosa.display

    y = wav.astype(np.float64)
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    db = librosa.amplitude_to_db(D, ref=np.max)
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(db, sr=sr, x_axis="time", y_axis="linear", hop_length=hop)
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Linear STFT: {p.stem}")
    plt.tight_layout()
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=120)
    plt.close()


def main() -> None:
    v = sys.version_info
    if v.major != 3 or v.minor < 9 or v.minor > 11:
        raise SystemExit("Python 3.9–3.11 рекомендуется (Coqui TTS).")

    ap = argparse.ArgumentParser(description="Синтез дообученного VITS + mel + STFT (PNG).")
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Папка runs/… с config.json (и чекпоинтом). По умолчанию — самый свежий в runs/.",
    )
    ap.add_argument(
        "--checkpoint", type=Path, default=None, help="Путь к .pth, если не best_model / последний"
    )
    ap.add_argument(
        "--text-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "synthesis_text.txt",
    )
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "finetuned_vits")
    ap.add_argument("--gpu", action="store_true", help="CUDA, если доступна")
    args = ap.parse_args()

    rdir = args.run_dir
    if rdir is None:
        runs = PROJECT_ROOT / "runs"
        if runs.is_dir():
            subs = [p for p in runs.iterdir() if p.is_dir()]
            rdir = max(subs, key=lambda p: p.stat().st_mtime) if subs else None
    if rdir is None or not rdir.is_dir():
        raise SystemExit("Укажите --run-dir=…/runs/vits_…")
    rdir = rdir.resolve()
    tts = _load_tts_for_run(
        rdir, args.checkpoint, use_gpu=args.gpu, progress_bar=True, cache=PROJECT_ROOT / "outputs" / ".tts_infer"
    )
    text = args.text_file.read_text(encoding="utf-8").replace("\ufeff", "")
    t0 = time.perf_counter()
    wav, sr = _synth_wav(tts, text)
    dt = time.perf_counter() - t0
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{rdir.name}_synthesis"
    wpath = out / f"{stem}.wav"
    sf.write(wpath, wav, sr)
    _save_mel(wav, sr, out / f"{stem}_mel.png")
    _save_linear_stft(wav, sr, out / f"{stem}_linear_stft.png")
    (out / "run_used.txt").write_text(str(rdir) + "\n", encoding="utf-8")
    print(f"OK за {dt:.2f} s: {wpath} | mel+linear: {out}")


if __name__ == "__main__":
    main()
