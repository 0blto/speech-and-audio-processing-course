import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _check_py() -> None:
    v = sys.version_info
    if v.major != 3 or v.minor < 9 or v.minor > 11:
        print(
            "Нужен Python 3.9–3.11 для пакета TTS. Создайте venv с подходящей версией.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _apply_hparams(
    config,
    lr_gen: float,
    lr_disc: float,
    batch_size: int,
    eval_batch: int,
    epochs: int,
) -> None:
    """VITS не использует config.lr: только lr_gen / lr_disc."""
    config.epochs = epochs
    config.batch_size = batch_size
    config.eval_batch_size = eval_batch
    config.lr_gen = lr_gen
    config.lr_disc = lr_disc


def _patch_vits_audio_loader(target_sr: int) -> None:
    """Patch VITS audio loader to avoid torchcodec issues and enforce target sample rate.

    В TTS 0.22 VitsDataset читает wav через `torchaudio.load` без ресемпла.
    Для VCTK (48k) + LJSpeech-VITS (22.05k) это ломает соответствие длительностей
    и часто даёт заметно замедленную речь на инференсе.
    """
    import numpy as np
    import soundfile as sf
    import torch
    from TTS.tts.models import vits as vits_mod

    def _safe_load_audio(file_path):
        wav_np, sr = sf.read(file_path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(np.asarray(wav_np)).transpose(0, 1).contiguous()
        sr = int(sr)
        if sr != int(target_sr):
            wav = torch.nn.functional.interpolate(
                wav.unsqueeze(0),
                scale_factor=float(target_sr) / float(sr),
                mode="linear",
                align_corners=False,
            ).squeeze(0)
            sr = int(target_sr)
        return wav, sr

    vits_mod.load_audio = _safe_load_audio


def main() -> None:
    _check_py()

    parser = argparse.ArgumentParser(
        description="VITS: fine-tune на одного диктора VCTK (2 пресета гиперпараметров A/B)"
    )
    parser.add_argument(
        "--exp",
        choices=["A", "B"],
        required=True,
        help=(
            "A: lr_gen=lr_disc=1.5e-4, batch=8 (агрессивнее, мельче батч). "
            "B: lr=4e-5, batch=20 (мягче, крупный батч). "
            "Суммарно при ~1800 эпох ожидаемо 8–20 ч на 3090 (см. README)."
        ),
    )
    parser.add_argument(
        "--vctk-single-root",
        type=Path,
        default=None,
        help="Корень из prepare_vctk_speaker.py (только txt/<spk> и wav48_*/<spk>).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1800,
        help="Число эпох. Уменьшите, если две сессии (A+B) уходят в >20 ч.",
    )
    parser.add_argument(
        "--pretrained",
        default="tts_models/en/ljspeech/vits",
        help="Имя предобученной модели Coqui (должна быть VITS).",
    )
    args = parser.parse_args()
    vroot = args.vctk_single_root
    if vroot is None:
        # первый VCTK_p* в data/
        cands = sorted((PROJECT_ROOT / "data").glob("VCTK_p*"))
        cands = [p for p in cands if p.is_dir() and (p / "txt").is_dir()]
        if len(cands) == 1:
            vroot = cands[0]
        if vroot is None:
            raise SystemExit(
                "Укажите --vctk-single-root или выполните:\n"
                "  python scripts/download_vctk.py\n"
                "  python scripts/prepare_vctk_speaker.py\n"
            )
    vroot = vroot.resolve()
    tdir = vroot / "txt"
    wdir = vroot / "wav48_silence_trimmed"
    if not tdir.is_dir() or not wdir.is_dir():
        raise SystemExit(
            f"В {vroot} должны быть каталоги txt/ и wav48_silence_trimmed/ (см. prepare_vctk_speaker.py)"
        )

    if args.exp == "A":
        lr_gen, lr_disc, batch_size = 1.5e-4, 1.5e-4, 8
    else:
        lr_gen, lr_disc, batch_size = 4e-5, 4e-5, 20
    eval_batch = min(16, batch_size)

    import torch
    from TTS.config import load_config
    from TTS.tts.configs.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.models import setup_model
    from TTS.utils.manage import ModelManager
    from trainer import Trainer, TrainerArgs

    if platform.system() == "Windows":
        import trainer.trainer as trainer_mod

        trainer_mod.remove_experiment_folder = lambda *_a, **_kw: None

    manager = ModelManager()
    model_path, config_path, _ = manager.download_model(args.pretrained)
    config = load_config(config_path)
    if getattr(config, "model", None) != "vits":
        raise SystemExit("Этот скрипт настроен только на model=vits. Укажите VITS в --pretrained.")

    if hasattr(config, "model_args") and getattr(config.model_args, "init_discriminator", None) is not True:
        config.model_args.init_discriminator = True

    dataset_config = BaseDatasetConfig(
        formatter="vctk",
        meta_file_train="",
        path=str(vroot),
        language="en",
    )
    config.datasets = [dataset_config]
    if hasattr(config, "use_phonemes") and not config.use_phonemes:
        print(
            "Предобученная LJSpeech-VITS ожидает фонемы; config.use_phonemes сейчас False — "
            "для согласованности включите eSpeak / espeak-ng в PATH (рекомендуется).",
            file=sys.stderr,
        )

    if hasattr(config, "audio") and config.audio is not None:
        config.audio.resample = True

    _apply_hparams(
        config,
        lr_gen=lr_gen,
        lr_disc=lr_disc,
        batch_size=batch_size,
        eval_batch=eval_batch,
        epochs=args.epochs,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"vits_vctk_{args.exp}_{ts}"
    runs = PROJECT_ROOT / "runs" / run_name
    runs.mkdir(parents=True, exist_ok=True)
    config.output_path = str(runs)
    config.run_name = run_name

    config.print_step = 200
    config.save_step = 10_000
    config.run_eval = True
    if hasattr(config, "test_delay_epochs"):
        config.test_delay_epochs = -1
    config.phoneme_cache_path = str(runs / "phoneme_cache")
    if hasattr(config, "eval_split_size"):
        config.eval_split_size = 0.05
    if hasattr(config, "eval_split_max_size"):
        config.eval_split_max_size = 80

    if platform.system() == "Windows":
        config.num_loader_workers = 0
        config.num_eval_loader_workers = 0
    if torch.cuda.is_available():
        if hasattr(config, "mixed_precision"):
            config.mixed_precision = True
    else:
        if hasattr(config, "mixed_precision"):
            config.mixed_precision = False

    target_sr = int(getattr(config.audio, "sample_rate", 22050)) if hasattr(config, "audio") else 22050
    _patch_vits_audio_loader(target_sr=target_sr)

    manifest = {
        "run_name": run_name,
        "vctk_single_root": str(vroot),
        "pretrained": args.pretrained,
        "exp": args.exp,
        "lr_gen": lr_gen,
        "lr_disc": lr_disc,
        "batch_size": batch_size,
        "eval_batch": eval_batch,
        "epochs": args.epochs,
        "restore_path": str(model_path),
    }
    (runs / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    if not train_samples:
        raise SystemExit("Пустой train: проверьте VCTK single-speaker и formatter vctk.")

    model = setup_model(config, train_samples + eval_samples)
    model.config.output_path = str(runs)

    train_args = TrainerArgs()
    train_args.restore_path = model_path

    trainer = Trainer(
        train_args,
        model.config,
        config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        parse_command_line_args=False,
    )

    print("Модель:", args.pretrained)
    print("Чекпоинт (init):", model_path)
    print("Train samples:", len(train_samples), "eval:", len(eval_samples))
    print("Выход (чекпоинты, config, TensorBoard):", runs)
    print("Кривые: tensorboard --logdir", runs, "  или  python scripts/export_curves_from_events.py", runs)
    print("---")
    trainer.fit()


if __name__ == "__main__":
    main()
