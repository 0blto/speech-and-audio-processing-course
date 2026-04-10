import argparse
import re
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _espeak_available() -> bool:
    return bool(shutil.which("espeak-ng") or shutil.which("espeak"))


def _load_tts(
    model_name: str,
    use_gpu: bool,
    progress_bar: bool,
    config_cache_dir: Path,
):
    from TTS.api import TTS
    from TTS.config import load_config
    from TTS.utils.manage import ModelManager

    if _espeak_available():
        return TTS(model_name=model_name, gpu=use_gpu, progress_bar=progress_bar)

    manager = ModelManager(progress_bar=progress_bar, verbose=False)
    model_path, config_path, _ = manager.download_model(model_name)
    config = load_config(config_path)

    use_ph = getattr(config, "use_phonemes", False)
    phon = getattr(config, "phonemizer", None)
    if use_ph and (phon in ("espeak", "gruut")):
        config.use_phonemes = False
        config.phonemizer = ""
        config.text_cleaner = "english_cleaners"
        safe = model_name.replace("/", "_").replace("\\", "_")
        config_cache_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = config_cache_dir / f"{safe}_no_phonemes.json"
        config.save_json(str(cfg_path))
        print(
            "  Примечание: eSpeak не найден в PATH — для этой модели отключены фонемы (use_phonemes=false). "
            "Чтобы использовать фонемы как в исходной модели, установите espeak-ng и добавьте его в PATH.",
            file=sys.stderr,
        )
        api = TTS(
            model_path=model_path,
            config_path=str(cfg_path),
            gpu=use_gpu,
            progress_bar=progress_bar,
        )
        api.model_name = model_name
        return api

    return TTS(model_name=model_name, gpu=use_gpu, progress_bar=progress_bar)


def _check_py() -> None:
    v = sys.version_info
    if v.major != 3 or v.minor < 9 or v.minor > 11:
        print("Нужен Python 3.9–3.11 для пакета TTS.", file=sys.stderr)
        raise SystemExit(1)


def split_into_chunks(text: str, max_chars: int = 180) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    buf = ""
    for p in parts:
        if not p:
            continue
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c]


def synthesize_long(tts, text: str) -> tuple[np.ndarray, int]:
    synth = tts.synthesizer
    if synth is None:
        raise RuntimeError("Модель не загружена (synthesizer is None).")

    chunks = split_into_chunks(text)
    pieces: list[np.ndarray] = []
    sr: int | None = None
    for ch in chunks:
        wav = synth.tts(text=ch, split_sentences=True)
        w = np.asarray(wav, dtype=np.float32)
        pieces.append(w)
        if sr is None:
            sr = int(getattr(synth, "output_sample_rate", 22050))
    if not pieces:
        return np.array([], dtype=np.float32), sr or 22050
    if len(pieces) == 1:
        return pieces[0], sr or 22050
    sil = np.zeros(int(0.15 * (sr or 22050)), dtype=np.float32)
    out: list[np.ndarray] = []
    for i, p in enumerate(pieces):
        out.append(p)
        if i < len(pieces) - 1:
            out.append(sil)
    return np.concatenate(out, axis=0), sr or 22050


def save_mel_png(wav: np.ndarray, sr: int, out_png: Path) -> None:
    try:
        import librosa
        import librosa.display
    except ImportError as e:
        raise SystemExit("Нужен librosa: pip install librosa") from e

    y = wav.astype(np.float64)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(out_png.stem)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=120)
    plt.close()


def main() -> None:
    _check_py()

    parser = argparse.ArgumentParser(description="Синтез и сравнение Tacotron2 vs VITS")
    parser.add_argument(
        "--text-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "synthesis_text.txt",
        help="Текст для синтеза",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "compare",
        help="Куда сохранить wav и mel PNG",
    )
    parser.add_argument("--gpu", action="store_true", help="Использовать CUDA, если есть")
    args = parser.parse_args()

    text = args.text_file.read_text(encoding="utf-8")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tts_cfg_cache = out / ".tts_runtime"

    use_gpu = args.gpu
    models = {
        "tacotron2": "tts_models/en/ljspeech/tacotron2-DDC",
        "vits": "tts_models/en/ljspeech/vits",
    }

    results: list[tuple[str, float, Path]] = []

    for short, name in models.items():
        print(f"Загрузка {short}…")
        tts = _load_tts(name, use_gpu=use_gpu, progress_bar=True, config_cache_dir=tts_cfg_cache)
        t0 = time.perf_counter()
        wav, sr = synthesize_long(tts, text)
        dt = time.perf_counter() - t0
        wav_path = out / f"{short}_synthesis.wav"
        sf.write(wav_path, wav, sr)
        mel_path = out / f"{short}_mel.png"
        save_mel_png(wav, sr, mel_path)
        results.append((short, dt, wav_path))
        print(f"  {short}: {dt:.2f} s -> {wav_path.name}")

    print("\nКратко (для отчёта):")
    for short, dt, wp in results:
        print(f"  - {short}: время {dt:.2f} с, файл {wp}")
    print("\nСлушайте wav и сравните разборчивость и естественность; mel-картинки — визуально сравните формантную структуру.")


if __name__ == "__main__":
    main()
