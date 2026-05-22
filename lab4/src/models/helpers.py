"""Общие утилиты для адаптеров моделей TTS.

Fallbacks:
    Если внешние зависимости недоступны, используются стандартные модули Python.
"""

# Подключаем стандартные модули
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
        Если замены не заданы, возвращается исходный текст.
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
    """Преобразует аудиообъект в плоский список float.

    Parameters:
        audio (object): Tensor, numpy array или список.

    Returns:
        list[float]: Плоский список сэмплов.

    Fallbacks:
        Если формат неизвестен, делается попытка итерироваться по объекту как по последовательности.
    """

    # Обрабатываем torch tensor при наличии методов detach и cpu
    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    if hasattr(audio, "tolist"):
        audio = audio.tolist()

    # Уплощаем вложенные списки
    if isinstance(audio, list):
        if audio and isinstance(audio[0], list):
            flattened = []
            for row in audio:
                flattened.extend(row)
            return [float(value) for value in flattened]
        return [float(value) for value in audio]

    # Преобразуем итерируемые объекты в список
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

    # Пытаемся использовать soundfile для сохранения
    try:
        import soundfile as sf

        sf.write(output_path, audio, sample_rate)
        return
    except ImportError:
        pass

    # Преобразуем сэмплы к int16
    samples = []
    for value in audio_to_samples(audio):
        clipped = max(-1.0, min(1.0, float(value)))
        samples.append(int(clipped * 32767.0))

    # Сохраняем wav стандартным модулем
    with wave.open(output_path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(int(sample).to_bytes(2, byteorder="little", signed=True) for sample in samples))


def read_audio_duration(output_path: str) -> float:
    """Читает длительность wav-файла в секундах.

    Parameters:
        output_path (str): Путь к wav-файлу.

    Returns:
        float: Длительность в секундах.

    Fallbacks:
        Если файл отсутствует или повреждён, возвращается 0.0.
    """

    # Пытаемся открыть wav как стандартный PCM-файл
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

