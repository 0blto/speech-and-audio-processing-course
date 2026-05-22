"""Общие утилиты для адаптеров моделей TTS.

Fallbacks:
    Если внешние зависимости недоступны, используются стандартные модули Python.
"""

# Подключаем стандартные модули
import array
import wave
from pathlib import Path


def prepare_text_common(config: dict, text: str) -> str:
    """Готовит текст по общим правилам из конфига.

    Parameters:
        config (dict): Конфигурация модели.
        text (str): Исходный текст.

    Returns:
        str: Подготовленный текст.

    Fallbacks:
        Если замены не заданы, возвращается нормализованный исходный текст.
    """

    # Нормализуем переносы строк и пробелы
    prepared = str(text).replace("\r\n", "\n").strip()

    # Применяем ручные замены текста из конфига
    replacements = config.get("text_replacements", {})
    if isinstance(replacements, dict):
        for source, target in replacements.items():
            prepared = prepared.replace(str(source), str(target))

    return prepared


def audio_to_samples(audio: object) -> list[float]:
    """Преобразует аудиообъект в плоский список mono-сэмплов.

    Parameters:
        audio (object): Tensor, numpy array или список сэмплов.

    Returns:
        list[float]: Плоский список сэмплов в формате float.

    Fallbacks:
        Если объект не поддерживает прямое преобразование, выполняется попытка итерации по нему как по последовательности.
    """

    # Переносим tensor на CPU и преобразуем к numpy при наличии нужных методов
    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()

    # Разворачиваем многомерный массив в одномерный
    if hasattr(audio, "reshape"):
        try:
            audio = audio.reshape(-1)
        except TypeError:
            pass

    # Преобразуем объект к обычным спискам Python
    if hasattr(audio, "tolist"):
        audio = audio.tolist()

    # Уплощаем вложенные списки в единый список float
    if isinstance(audio, list):
        flattened: list[float] = []
        stack = list(audio)
        while stack:
            value = stack.pop(0)
            if isinstance(value, list):
                stack = list(value) + stack
                continue
            flattened.append(float(value))
        return flattened

    # Преобразуем итерируемый объект в список float
    return [float(value) for value in audio]


def save_audio_data(output_path: str, audio: object, sample_rate: int) -> None:
    """Сохраняет аудиообъект в wav-файл.

    Parameters:
        output_path (str): Путь к итоговому wav.
        audio (object): Tensor, numpy array или список сэмплов.
        sample_rate (int): Частота дискретизации.

    Returns:
        None: Ничего не возвращает.

    Fallbacks:
        Если soundfile недоступен, используется стандартный модуль wave.
    """

    # Создаём родительскую папку для выходного файла
    ensure_parent_dir(output_path)

    # Приводим аудио к плоскому списку mono-сэмплов
    samples_float = audio_to_samples(audio)

    # Пытаемся сохранить wav через soundfile
    try:
        import soundfile as sf

        sf.write(output_path, samples_float, sample_rate)
        return
    except ImportError:
        pass

    # Преобразуем float-сэмплы в int16 для стандартного wave writer
    samples = array.array("h")
    for value in samples_float:
        clipped = max(-1.0, min(1.0, float(value)))
        samples.append(int(clipped * 32767.0))

    # Сохраняем wav стандартным модулем wave
    with wave.open(output_path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def read_audio_duration(output_path: str) -> float:
    """Читает длительность wav-файла в секундах.

    Parameters:
        output_path (str): Путь к wav-файлу.

    Returns:
        float: Длительность в секундах.

    Fallbacks:
        Если файл отсутствует или повреждён, возвращается 0.0.
    """

    # Пытаемся открыть wav и вычислить длительность по числу фреймов
    try:
        with wave.open(output_path, "rb") as handle:
            frame_count = handle.getnframes()
            frame_rate = handle.getframerate()
        if frame_rate <= 0:
            return 0.0
        return round(frame_count / frame_rate, 3)
    except (FileNotFoundError, OSError, wave.Error):
        return 0.0


def ensure_parent_dir(output_path: str) -> None:
    """Создаёт родительскую папку для файла.

    Parameters:
        output_path (str): Путь к выходному файлу.

    Returns:
        None: Ничего не возвращает.

    Fallbacks:
        Если папка уже существует, ошибка не возникает.
    """

    # Создаём каталог результата
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
