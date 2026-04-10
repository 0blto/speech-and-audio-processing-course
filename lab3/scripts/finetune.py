import argparse
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


def main() -> None:
    _check_py()

    parser = argparse.ArgumentParser(description="Дообучение Coqui TTS (1 эпоха)")
    parser.add_argument(
        "--model",
        choices=["tacotron2", "vits"],
        required=True,
        help="tacotron2 — классическая модель; vits — современная end-to-end",
    )
    parser.add_argument(
        "--exp",
        choices=["A", "B"],
        required=True,
        help="A: lr=1e-4, batch=8; B: lr=2.5e-5, batch=20",
    )
    parser.add_argument(
        "--ljspeech-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "LJSpeech-1.1",
        help="Каталог LJSpeech-1.1",
    )
    args = parser.parse_args()
    ljspeech_root = args.ljspeech_root.resolve()

    meta = ljspeech_root / "metadata_subset.csv"
    if not meta.is_file():
        raise SystemExit(
            f"Нет {meta}. Выполните: python scripts/prepare_subset.py\n"
            f"и убедитесь, что LJSpeech распакован в {ljspeech_root}"
        )

    model_names = {
        "tacotron2": "tts_models/en/ljspeech/tacotron2-DDC",
        "vits": "tts_models/en/ljspeech/vits",
    }
    pretrained_name = model_names[args.model]

    if args.exp == "A":
        lr, batch_size = 1e-4, 8
    else:
        lr, batch_size = 2.5e-5, 20

    import torch
    from TTS.config import load_config

    try:
        from TTS.tts.configs.shared_configs import BaseDatasetConfig
    except ImportError:
        from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.models import setup_model
    from TTS.utils.manage import ModelManager
    from trainer import Trainer, TrainerArgs

    manager = ModelManager()
    model_path, config_path, _ = manager.download_model(pretrained_name)
    config = load_config(config_path)

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata_subset.csv",
        path=str(ljspeech_root),
    )
    config.datasets = [dataset_config]

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.model}_finetune_exp{args.exp}_lr{lr}_bs{batch_size}_{ts}"
    runs = PROJECT_ROOT / "runs" / run_name
    runs.mkdir(parents=True, exist_ok=True)
    config.output_path = str(runs)
    config.run_name = run_name

    config.epochs = 1
    config.batch_size = batch_size
    config.eval_batch_size = min(8, batch_size)
    config.lr = lr
    config.print_step = 10
    config.save_step = 50
    config.phoneme_cache_path = str(runs / "phoneme_cache")
    if hasattr(config, "eval_split_size"):
        config.eval_split_size = 0.05
    if hasattr(config, "eval_split_max_size"):
        config.eval_split_max_size = 50

    if platform.system() == "Windows":
        config.num_loader_workers = 0
        config.num_eval_loader_workers = 0
    if not torch.cuda.is_available():
        if hasattr(config, "mixed_precision"):
            config.mixed_precision = False

    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

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

    print(f"Модель: {pretrained_name}")
    print(f"Checkpoint: {model_path}")
    print(f"Выход: {runs}")
    print("TensorBoard: tensorboard --logdir", runs)
    print("---")
    trainer.fit()


if __name__ == "__main__":
    main()
