"""Адаптер для Silero TTS V5.

Fallbacks:
    Если silero или torch не установлены, адаптер завершится RuntimeError с понятным сообщением.
"""

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Подключаем общие утилиты
from models.helpers import prepare_text_common
from models.helpers import concat_audio_segments
from models.helpers import read_audio_duration
from models.helpers import save_audio_data
from models.helpers import split_text_for_tts


def validate_config(config: dict) -> None:
    """Проверяет конфигурацию модели Silero V5.

    Parameters:
        config (dict): Конфигурация модели.

    Returns:
        None: Ничего не возвращает.

    Fallbacks:
        При отсутствии обязательных полей выбрасывается ValueError.
    """

    # Проверяем обязательные ключи
    required = ["id", "model_type", "language", "speaker_model", "speaker_name", "sample_rate"]
    for field_name in required:
        if not str(config.get(field_name, "")).strip():
            raise ValueError(f"Для модели {config.get('id', '<unknown>')} нужно поле {field_name}")


def load_model(config: dict) -> object:
    """Загружает Silero V5.

    Parameters:
        config (dict): Конфигурация модели.

    Returns:
        object: Загруженная модель Silero.

    Fallbacks:
        Если пакет silero недоступен, используется torch.hub.load.
    """

    # Импортируем torch в момент загрузки
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Для Silero V5 нужен torch 2.0+") from error

    # Сначала пробуем pip-пакет silero
    try:
        from silero import silero_tts

        model, _ = silero_tts(
            language=str(config["language"]),
            speaker=str(config["speaker_model"]),
        )
    except ImportError:
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language=str(config["language"]),
            speaker=str(config["speaker_model"]),
        )

    # Переводим модель на нужное устройство
    model.to(str(config.get("device", "cpu")))
    return model


def prepare_text(config: dict, text: str) -> str:
    """Подготавливает текст для Silero V5.

    Parameters:
        config (dict): Конфигурация модели.
        text (str): Исходный текст.

    Returns:
        str: Подготовленный текст.

    Fallbacks:
        Модель использует собственные автоударения, поэтому спецобработка минимальна.
    """

    # Возвращаем только общую нормализацию
    return prepare_text_common(config, text)


def synthesize(model_handle: object, config: dict, text: str, output_path: str) -> dict:
    """Синтезирует wav через Silero V5.

    Parameters:
        model_handle (object): Загруженная модель.
        config (dict): Конфигурация модели.
        text (str): Подготовленный текст.
        output_path (str): Путь к выходному wav.

    Returns:
        dict: Метаданные результата синтеза.

    Fallbacks:
        Если save_wav недоступен, используется apply_tts и ручное сохранение.
    """

    # Получаем параметры выбранного голоса
    speaker_name = str(config["speaker_name"])
    sample_rate = int(config["sample_rate"])
    max_text_length = int(config.get("max_text_length", 950))
    text_chunks = split_text_for_tts(text, max_text_length)

    # Для длинного текста синтезируем куски отдельно и склеиваем в один wav.
    if len(text_chunks) > 1:
        audio_segments = []
        chunk_iterator = text_chunks
        if tqdm is not None:
            chunk_iterator = tqdm(text_chunks, desc=f"{config.get('id', 'silero_v5')} chunks", unit="chunk")
        for chunk in chunk_iterator:
            audio_segments.append(model_handle.apply_tts(text=chunk, speaker=speaker_name, sample_rate=sample_rate))
        save_audio_data(output_path, concat_audio_segments(audio_segments, sample_rate), sample_rate)
        return {
            "audio_path": output_path,
            "audio_duration_sec": read_audio_duration(output_path),
            "sample_rate": sample_rate,
            "status": "generated",
            "error_message": "",
        }

    # Пытаемся сохранить wav штатным методом модели
    try:
        model_handle.save_wav(text=text, speaker=speaker_name, sample_rate=sample_rate, audio_path=output_path)
    except TypeError:
        audio = model_handle.apply_tts(text=text, speaker=speaker_name, sample_rate=sample_rate)
        save_audio_data(output_path, audio, sample_rate)

    return {
        "audio_path": output_path,
        "audio_duration_sec": read_audio_duration(output_path),
        "sample_rate": sample_rate,
        "status": "generated",
        "error_message": "",
    }
