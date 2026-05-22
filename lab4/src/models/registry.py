"""Реестр стандартизированных адаптеров моделей TTS.

Fallbacks:
    Если запрошенный тип модели не зарегистрирован, выбрасывается ValueError.
"""

# Подключаем стандартные модули
import importlib

# Храним соответствие типа модели и модуля адаптера
MODEL_MODULES = {
    "chatterbox": "models.chatterbox_model",
    "f5": "models.f5_tts_model",
    "espeech": "models.espeech_model",
    "silero_v5": "models.silero_v5_model",
    "silero_v4": "models.silero_v4_model",
}


def get_model_adapter(model_type: str) -> object:
    """Возвращает модуль-адаптер по типу модели.

    Parameters:
        model_type (str): Ключ типа модели.

    Returns:
        object: Импортированный модуль адаптера.

    Fallbacks:
        Если ключ отсутствует, выбрасывается ValueError.
    """

    # Проверяем, что тип модели зарегистрирован
    if model_type not in MODEL_MODULES:
        raise ValueError(f"Неподдерживаемый model_type: {model_type}")

    # Импортируем модуль адаптера по имени
    return importlib.import_module(MODEL_MODULES[model_type])


def load_model_once(config: dict, cache: dict) -> object:
    """Загружает модель один раз и переиспользует её по id.

    Parameters:
        config (dict): Конфигурация модели.
        cache (dict): Кэш загруженных моделей.

    Returns:
        object: Хэндл загруженной модели.

    Fallbacks:
        Если модели ещё нет в кэше, выполняется загрузка через адаптер.
    """

    # Проверяем наличие идентификатора модели
    model_id = str(config.get("id", "")).strip()
    if not model_id:
        raise ValueError("В конфиге модели отсутствует обязательное поле id")

    # Возвращаем кэшированный хэндл при наличии
    if model_id in cache:
        return cache[model_id]

    # Загружаем модель через зарегистрированный адаптер
    adapter = get_model_adapter(str(config.get("model_type", "")).strip())
    model_handle = adapter.load_model(config)
    cache[model_id] = model_handle
    return model_handle

