from pathlib import Path
from TTS.utils.downloaders import download_ljspeech


ROOT_PATH = Path(__file__).parent.parent
download_ljspeech(ROOT_PATH / "recipes/ljspeech/")
