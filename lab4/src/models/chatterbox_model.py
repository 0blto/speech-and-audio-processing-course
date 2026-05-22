"""Адаптер для Chatterbox Multilingual.

Fallbacks:
    Если пакет chatterbox-tts не установлен, адаптер завершится RuntimeError с понятным сообщением.
"""

# Подключаем общие утилиты
import inspect

from models.helpers import prepare_text_common
from models.helpers import read_audio_duration
from models.helpers import save_audio_data


def validate_config(config: dict) -> None:
    """Проверяет конфигурацию модели Chatterbox.

    Parameters:
        config (dict): Конфигурация модели.

    Returns:
        None: Ничего не возвращает.

    Fallbacks:
        При отсутствии обязательных полей выбрасывается ValueError.
    """

    # Проверяем обязательные ключи
    required = ["id", "model_type", "language_id", "sample_rate"]
    for field_name in required:
        if not str(config.get(field_name, "")).strip():
            raise ValueError(f"Для модели {config.get('id', '<unknown>')} нужно поле {field_name}")


def load_model(config: dict) -> object:
    """Загружает Chatterbox Multilingual.

    Parameters:
        config (dict): Конфигурация модели.

    Returns:
        object: Загруженный объект модели.

    Fallbacks:
        Если пакет не установлен, выбрасывается RuntimeError.
    """

    # Импортируем библиотеку только в момент загрузки
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except ImportError as error:
        raise RuntimeError(
            "Для Chatterbox нужен пакет chatterbox-tts и рекомендуемый Python 3.11"
        ) from error

    # Загружаем multilingual модель на нужное устройство
    device = str(config.get("device", "cpu"))
    t3_model = str(config.get("t3_model", "")).strip()
    from_pretrained = ChatterboxMultilingualTTS.from_pretrained
    parameter_names = set(inspect.signature(from_pretrained).parameters)

    kwargs = {}
    if "device" in parameter_names:
        kwargs["device"] = device
    if t3_model and "t3_model" in parameter_names:
        kwargs["t3_model"] = t3_model

    try:
        return from_pretrained(**kwargs)
    except TypeError:
        # Older/newer chatterbox-tts releases may expose a narrower loader signature.
        return from_pretrained()


def prepare_text(config: dict, text: str) -> str:
    """Подготавливает текст для Chatterbox.

    Parameters:
        config (dict): Конфигурация модели.
        text (str): Исходный текст.

    Returns:
        str: Подготовленный текст.

    Fallbacks:
        Если дополнительные правила не заданы, возвращается нормализованный текст.
    """

    # Применяем общие правила без спецобработки ударений
    return prepare_text_common(config, text)


def synthesize(model_handle: object, config: dict, text: str, output_path: str) -> dict:
    """Синтезирует wav через Chatterbox.

    Parameters:
        model_handle (object): Загруженная модель.
        config (dict): Конфигурация модели.
        text (str): Подготовленный текст.
        output_path (str): Путь к выходному wav.

    Returns:
        dict: Метаданные результата синтеза.

    Fallbacks:
        Если audio prompt не задан, используется дефолтный голос модели.
    """

    # Получаем параметры синтеза из конфига
    language_id = str(config.get("language_id", "ru"))
    audio_prompt_path = str(config.get("ref_audio_path", "")).strip()

    # Запускаем генерацию с voice prompt или без него
    if audio_prompt_path:
        wav = model_handle.generate(text, language_id=language_id, audio_prompt_path=audio_prompt_path)
    else:
        wav = model_handle.generate(text, language_id=language_id)

    # Сохраняем аудио в wav
    save_audio_data(output_path, wav, int(config.get("sample_rate", getattr(model_handle, "sr", 24000))))

    return {
        "audio_path": output_path,
        "audio_duration_sec": read_audio_duration(output_path),
        "sample_rate": int(config.get("sample_rate", getattr(model_handle, "sr", 24000))),
        "status": "generated",
        "error_message": "",
    }
