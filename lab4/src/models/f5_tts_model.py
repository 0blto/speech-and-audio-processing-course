"""Адаптер для F5-TTS_RUSSIAN.

Fallbacks:
    Если пакет f5-tts не установлен, адаптер завершится RuntimeError с понятным сообщением.
"""

# Подключаем общие утилиты
from models.helpers import prepare_text_common
from models.helpers import read_audio_duration
from models.helpers import save_audio_data


def validate_config(config: dict) -> None:
    """Проверяет конфигурацию модели F5-TTS.

    Parameters:
        config (dict): Конфигурация модели.

    Returns:
        None: Ничего не возвращает.

    Fallbacks:
        При отсутствии обязательных полей выбрасывается ValueError.
    """

    # Проверяем обязательные поля для reference-based модели
    required = ["id", "model_type", "checkpoint_path", "vocab_path", "ref_audio_path", "ref_text"]
    for field_name in required:
        if not str(config.get(field_name, "")).strip():
            raise ValueError(f"Для модели {config.get('id', '<unknown>')} нужно поле {field_name}")


def load_model(config: dict) -> object:
    """Загружает F5-TTS и возвращает хэндл с моделью.

    Parameters:
        config (dict): Конфигурация модели.

    Returns:
        object: Словарь с model и vocoder.

    Fallbacks:
        Если нужные модули не установлены, выбрасывается RuntimeError.
    """

    # Импортируем F5-TTS только при реальном использовании
    try:
        from f5_tts.infer.utils_infer import load_model
        from f5_tts.infer.utils_infer import load_vocoder
        from f5_tts.model import DiT
    except ImportError as error:
        raise RuntimeError(
            "Для F5-TTS_RUSSIAN нужны пакеты f5-tts, torch и torchaudio"
        ) from error

    # Готовим конфиг DiT по умолчанию для F5
    model_cfg = config.get(
        "model_cfg",
        {
            "dim": 1024,
            "depth": 22,
            "heads": 16,
            "ff_mult": 2,
            "text_dim": 512,
            "conv_layers": 4,
        },
    )

    # Загружаем модель и вокодер
    model = load_model(DiT, model_cfg, str(config["checkpoint_path"]), vocab_file=str(config["vocab_path"]))
    vocoder = load_vocoder()
    return {"model": model, "vocoder": vocoder}


def prepare_text(config: dict, text: str) -> str:
    """Подготавливает текст для F5-TTS.

    Parameters:
        config (dict): Конфигурация модели.
        text (str): Исходный текст.

    Returns:
        str: Подготовленный текст.

    Fallbacks:
        Если автоакцентирование недоступно, остаётся только ручная разметка через символ +.
    """

    # Применяем общую нормализацию текста
    prepared = prepare_text_common(config, text)

    # Опционально применяем RUAccent
    accent_mode = str(config.get("accent_mode", "manual_plus"))
    if accent_mode != "ruaccent":
        return prepared

    try:
        from ruaccent import RUAccent
    except ImportError:
        return prepared

    # Автоматически расставляем ударения только если их ещё нет
    if "+" in prepared:
        return prepared
    accentizer = RUAccent()
    accentizer.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
    return accentizer.process_all(prepared)


def synthesize(model_handle: object, config: dict, text: str, output_path: str) -> dict:
    """Синтезирует wav через F5-TTS.

    Parameters:
        model_handle (object): Хэндл с model и vocoder.
        config (dict): Конфигурация модели.
        text (str): Подготовленный текст.
        output_path (str): Путь к выходному wav.

    Returns:
        dict: Метаданные результата синтеза.

    Fallbacks:
        Если direct API недоступен, выбрасывается RuntimeError.
    """

    # Импортируем API-инференс только в момент вызова
    try:
        from f5_tts.infer.utils_infer import infer_process
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text
    except ImportError as error:
        raise RuntimeError("Не удалось импортировать infer_process из f5-tts") from error

    # Извлекаем объекты модели и параметры генерации
    model = model_handle["model"]
    vocoder = model_handle["vocoder"]
    ref_audio_path = str(config["ref_audio_path"])
    ref_text = str(config["ref_text"])
    speed = float(config.get("speed", 1.0))
    nfe_step = int(config.get("nfe_step", 32))
    cross_fade_duration = float(config.get("cross_fade_duration", 0.15))

    # Готовим референс перед инференсом
    ref_audio_ready, ref_text_ready = preprocess_ref_audio_text(ref_audio_path, ref_text)

    # Выполняем инференс модели
    final_wave, final_sample_rate, _ = infer_process(
        ref_audio_ready,
        ref_text_ready,
        text,
        model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
    )

    # Сохраняем аудио в wav
    save_audio_data(output_path, final_wave, int(final_sample_rate))

    return {
        "audio_path": output_path,
        "audio_duration_sec": read_audio_duration(output_path),
        "sample_rate": int(final_sample_rate),
        "status": "generated",
        "error_message": "",
    }

