from pathlib import Path

# Trainer: Where the ✨️ happens.
# TrainingArgs: Defines the set of arguments of the Trainer.
from trainer import Trainer, TrainerArgs

# VitsConfig: all model related values for training, validating and testing.
from TTS.tts.configs.vits_config import VitsConfig

# BaseDatasetConfig: defines name, formatter and path of the dataset.
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# we use the same path as this script as our training folder.
ROOT_PATH = Path(__file__).parent.parent
output_path = ROOT_PATH / "outputs"

# DEFINE DATASET CONFIG
# Set LJSpeech as our target dataset and define its path.
# You can also use a simple Dict to define the dataset and pass it to your custom formatter.
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="metadata.csv",
    path=str(ROOT_PATH / "recipes/ljspeech/LJSpeech-1.1"),
)
audio_config = VitsAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None,
)

# INITIALIZE THE TRAINING CONFIGURATION
# Configure the model. Every config class inherits the BaseTTSConfig.
config = VitsConfig(
    audio=audio_config,
    run_name="vits_ljspeech",
    batch_size=4,
    eval_batch_size=4,
    batch_group_size=5,
    num_loader_workers=2,
    num_eval_loader_workers=1,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=2,
    text_cleaner="english_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=str(output_path / "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=1,
    print_eval=True,
    mixed_precision=False,
    output_path=str(output_path),
    datasets=[dataset_config],
    cudnn_benchmark=False,
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# If characters are not defined in the config, default characters are passed to the config
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# INITIALIZE THE MODEL
# Models take a config object and a speaker manager as input
# Config defines the details of the model like the number of layers, the size of the embedding, etc.
# Speaker manager is used by multi-speaker models.
model = Vits(config, ap, tokenizer, speaker_manager=None)

# INITIALIZE THE TRAINER
# Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# distributed training, etc.
trainer = Trainer(
    TrainerArgs(),
    config,
    str(output_path),
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
