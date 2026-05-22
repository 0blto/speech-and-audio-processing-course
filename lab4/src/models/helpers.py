"""Общие утилиты для адаптеров моделей TTS.

Fallbacks:
    Если внешние зависимости недоступны, используются стандартные модули Python.
"""

# Подключаем стандартные модули
import array
import re
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


def split_text_for_tts(text: str, max_length: int) -> list[str]:
    """Разбивает длинный текст на куски, удобные для TTS.

    Parameters:
        text (str): Подготовленный текст для синтеза.
        max_length (int): Максимальная длина одного куска.

    Returns:
        list[str]: Список кусков текста в исходном порядке.

    Fallbacks:
        Если текст нельзя разбить по абзацам, предложениям или словам,
        используется жёсткое разбиение по символам.
    """

    normalized = str(text).strip()
    if not normalized:
        return []

    if max_length <= 0 or len(normalized) <= max_length:
        return [normalized]

    chunks: list[str] = []

    def flush_buffer(buffer: str) -> str:
        cleaned = buffer.strip()
        if cleaned:
            chunks.append(cleaned)
        return ""

    def split_hard(fragment: str) -> None:
        remaining = fragment.strip()
        while remaining:
            if len(remaining) <= max_length:
                chunks.append(remaining)
                return
            split_index = remaining.rfind(" ", 0, max_length + 1)
            if split_index <= 0:
                split_index = max_length
            chunks.append(remaining[:split_index].strip())
            remaining = remaining[split_index:].strip()

    # Сначала пробуем абзацы, затем предложения внутри них.
    paragraphs = [part.strip() for part in normalized.split("\n") if part.strip()]
    for paragraph in paragraphs:
        if len(paragraph) <= max_length:
            chunks.append(paragraph)
            continue

        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?…])\s+", paragraph)
            if part.strip()
        ]
        buffer = ""
        for sentence in sentences:
            if len(sentence) > max_length:
                buffer = flush_buffer(buffer)
                split_hard(sentence)
                continue

            candidate = sentence if not buffer else f"{buffer} {sentence}"
            if len(candidate) <= max_length:
                buffer = candidate
                continue

            buffer = flush_buffer(buffer)
            buffer = sentence

        flush_buffer(buffer)

    return chunks


def concat_audio_segments(segments: list[object], sample_rate: int, pause_ms: int = 120) -> list[float]:
    """Склеивает несколько аудиосегментов в один поток mono-сэмплов.

    Parameters:
        segments (list[object]): Список аудиосегментов.
        sample_rate (int): Частота дискретизации итогового wav.
        pause_ms (int): Небольшая пауза между сегментами в миллисекундах.

    Returns:
        list[float]: Плоский список итоговых сэмплов.

    Fallbacks:
        Если сегменты пустые, возвращается пустой список.
    """

    if not segments:
        return []

    pause_samples = max(0, int(sample_rate * pause_ms / 1000))
    silence = [0.0] * pause_samples
    merged: list[float] = []

    for index, segment in enumerate(segments):
        merged.extend(audio_to_samples(segment))
        if index != len(segments) - 1 and pause_samples:
            merged.extend(silence)

    return merged


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
    except Exception:
        # Если soundfile установлен, но не может записать wav в текущем окружении,
        # откатываемся на стандартный wave writer.
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
