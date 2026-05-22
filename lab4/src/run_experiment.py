"""Запускает генерацию аудио и собирает сводные результаты лабораторной 4.

Fallbacks:
    Если зависимости конкретной модели не установлены, скрипт завершает только проблемную модель с понятной ошибкой.
"""

# Подключаем стандартные модули
import argparse
import csv
import json
import sys
from pathlib import Path

# Подключаем локальные константы
from constants import AUDIO_DIR
from constants import DATA_DIR
from constants import GROUP_LABELS
from constants import MANUAL_SCORE_FIELDS
from constants import MANUAL_SCORES_PATH
from constants import MODELS_PATH
from constants import RESULTS_DIR
from constants import RUNS_PATH
from constants import SUMMARY_CSV_PATH
from constants import SUMMARY_JSON_PATH
from constants import TEXT_GROUPS

# Подключаем реестр моделей
from models.registry import get_model_adapter
from models.registry import load_model_once


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки.

    Parameters:
        Нет.

    Returns:
        argparse.Namespace: Объект с аргументами CLI.

    Fallbacks:
        При неверном наборе флагов argparse завершает программу с сообщением об ошибке.
    """

    # Описываем CLI
    parser = argparse.ArgumentParser(description="Лабораторная 4: сравнение локальных TTS")
    parser.add_argument("--generate", action="store_true", help="Сгенерировать аудио и шаблон manual_scores.csv")
    parser.add_argument("--summarize", action="store_true", help="Посчитать summary.csv и summary.json из manual_scores.csv")
    parser.add_argument("--model", type=str, default="", help="Запустить только одну модель по её id")
    parser.add_argument("--force", action="store_true", help="Перезаписать уже существующие wav-файлы")
    args = parser.parse_args()

    # Проверяем, что пользователь выбрал действие
    if not args.generate and not args.summarize:
        parser.error("Нужно указать хотя бы один флаг: --generate или --summarize")

    return args


def ensure_directories() -> None:
    """Создаёт рабочие директории проекта.

    Parameters:
        Нет.

    Returns:
        None: Ничего не возвращает.

    Fallbacks:
        Если директории уже существуют, метод pathlib не вызывает ошибку.
    """

    # Создаём обязательные папки
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def load_models(models_path: Path) -> list[dict]:
    """Загружает список моделей из json-файла.

    Parameters:
        models_path (Path): Путь к models.json.

    Returns:
        list[dict]: Список описаний моделей.

    Fallbacks:
        Если файл отсутствует или пустой, выбрасывается понятная ошибка.
    """

    # Проверяем наличие файла конфигурации
    if not models_path.exists():
        raise FileNotFoundError(
            f"Не найден файл моделей: {models_path}. "
            "Создайте его или используйте подготовленный lab4/data/models.json"
        )

    # Загружаем json с моделями
    payload = json.loads(models_path.read_text(encoding="utf-8"))

    # Проверяем формат корневой структуры
    if not isinstance(payload, list) or not payload:
        raise ValueError("models.json должен содержать непустой список моделей")

    return payload


def normalize_text_groups() -> list[dict]:
    """Преобразует текстовые группы в плоский список элементов.

    Parameters:
        Нет.

    Returns:
        list[dict]: Список записей вида group/item_id/text.

    Fallbacks:
        Если одна из групп пуста, она просто не добавляет записи.
    """

    # Разворачиваем группы в единый список
    rows = []
    for group_name, items in TEXT_GROUPS.items():
        for index, text in enumerate(items, start=1):
            rows.append(
                {
                    "group": group_name,
                    "item_id": f"{group_name}_{index:02d}",
                    "text": text,
                }
            )

    return rows


def sanitize_model_id(value: str) -> str:
    """Готовит идентификатор модели для имени файла.

    Parameters:
        value (str): Исходный id модели.

    Returns:
        str: Безопасная строка для имени файла.

    Fallbacks:
        Если после очистки строка пуста, возвращается литерал model.
    """

    # Оставляем только безопасные символы
    cleaned = "".join(symbol if symbol.isalnum() or symbol in "-_" else "_" for symbol in value)
    return cleaned or "model"


def resolve_audio_path(model_id: str, group: str, item_id: str) -> Path:
    """Строит путь к итоговому wav-файлу.

    Parameters:
        model_id (str): Идентификатор модели.
        group (str): Название группы текстов.
        item_id (str): Идентификатор элемента.

    Returns:
        Path: Путь к wav-файлу.

    Fallbacks:
        Если каталог не существует, он будет создан заранее отдельной функцией.
    """

    # Формируем единый шаблон имени wav
    file_name = f"{sanitize_model_id(model_id)}_{group}_{item_id}.wav"
    return AUDIO_DIR / file_name


def synthesize_item(model_config: dict, text_row: dict, force: bool, model_cache: dict) -> dict:
    """Генерирует один wav и возвращает метаданные запуска.

    Parameters:
        model_config (dict): Конфигурация модели.
        text_row (dict): Описание одного текста.
        force (bool): Нужно ли перезаписывать существующий wav.
        model_cache (dict): Кэш загруженных моделей.

    Returns:
        dict: Метаданные одной генерации.

    Fallbacks:
        Если wav уже существует и force=False, генерация пропускается.
    """

    # Готовим путь к итоговому wav
    audio_path = resolve_audio_path(str(model_config["id"]), str(text_row["group"]), str(text_row["item_id"]))
    status = "skipped"
    error_message = ""
    duration_sec = 0.0
    sample_rate = ""

    # Получаем адаптер модели и проверяем конфиг
    adapter = get_model_adapter(str(model_config["model_type"]))
    adapter.validate_config(model_config)

    # Генерируем wav только при необходимости
    if force or not audio_path.exists():
        print(
            f"[generate] model={model_config['id']} "
            f"item={text_row['item_id']} "
            f"group={text_row['group']} "
            f"type={model_config['model_type']}"
        )
        try:
            model_handle = load_model_once(model_config, model_cache)
            prepared_text = adapter.prepare_text(model_config, str(text_row["text"]))
            result = adapter.synthesize(model_handle, model_config, prepared_text, str(audio_path))
            status = str(result.get("status", "generated"))
            error_message = str(result.get("error_message", ""))
            duration_sec = float(result.get("audio_duration_sec", 0.0))
            sample_rate = result.get("sample_rate", "")
            print(f"[ok] model={model_config['id']} item={text_row['item_id']} file={audio_path.name}")
        except Exception as error:
            status = "failed"
            error_message = str(error)
            print(f"[error] model={model_config['id']} item={text_row['item_id']} message={error_message}", file=sys.stderr)

    # Возвращаем метаданные запуска для аудио или ошибки
    audio_path_value = str(audio_path.relative_to(RESULTS_DIR.parent)) if audio_path.exists() else ""
    return {
        "model_id": str(model_config["id"]),
        "model_name": str(model_config.get("name", model_config["id"])),
        "model_type": str(model_config["model_type"]),
        "group": str(text_row["group"]),
        "item_id": str(text_row["item_id"]),
        "text": str(text_row["text"]),
        "audio_path": audio_path_value,
        "audio_duration_sec": duration_sec,
        "sample_rate": sample_rate,
        "status": status,
        "error_message": error_message,
        "reference_used": "yes" if model_config.get("needs_ref_audio") else "no",
    }


def build_manual_rows(run_rows: list[dict]) -> list[dict]:
    """Строит строки шаблона manual_scores.csv.

    Parameters:
        run_rows (list[dict]): Метаданные генераций.

    Returns:
        list[dict]: Строки для ручной оценки.

    Fallbacks:
        Если часть аудио не сгенерирована, строки всё равно создаются.
    """

    # Переносим автоматические поля и оставляем ручные пустыми
    rows = []
    for item in run_rows:
        rows.append(
            {
                "model_id": item["model_id"],
                "model_name": item["model_name"],
                "model_type": item["model_type"],
                "group": item["group"],
                "item_id": item["item_id"],
                "text": item["text"],
                "audio_path": item["audio_path"],
                "audio_duration_sec": item["audio_duration_sec"],
                "sample_rate": item["sample_rate"],
                "reference_used": item["reference_used"],
                "generation_status": item["status"],
                "latency_sec": "",
                "has_error": "",
                "error_notes": item.get("error_message", ""),
                "stress_notes": "",
            }
        )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Записывает список словарей в csv-файл.

    Parameters:
        path (Path): Путь к csv-файлу.
        fieldnames (list[str]): Порядок колонок.
        rows (list[dict]): Данные для записи.

    Returns:
        None: Ничего не возвращает.

    Fallbacks:
        Если записей нет, создаётся только заголовок.
    """

    # Открываем csv в utf-8 и пишем заголовок
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_manual_row_key(row: dict) -> tuple[str, str, str]:
    """Строит стабильный ключ строки manual_scores.csv.

    Parameters:
        row (dict): Строка manual_scores.csv.

    Returns:
        tuple[str, str, str]: Ключ вида model_id/group/item_id.

    Fallbacks:
        Если часть полей отсутствует, в ключ попадают пустые строки.
    """

    return (
        str(row.get("model_id", "")).strip(),
        str(row.get("group", "")).strip(),
        str(row.get("item_id", "")).strip(),
    )


def merge_manual_rows(existing_rows: list[dict], new_rows: list[dict], force: bool) -> list[dict]:
    """Объединяет старые и новые строки manual_scores.csv.

    Parameters:
        existing_rows (list[dict]): Уже сохранённые строки manual_scores.csv.
        new_rows (list[dict]): Новые строки для текущего запуска.
        force (bool): Нужно ли полностью заменить строки затронутых моделей.

    Returns:
        list[dict]: Итоговый набор строк для сохранения.

    Fallbacks:
        Если новых строк нет, возвращает существующие строки без изменений.
    """

    if not new_rows:
        return existing_rows

    target_model_ids = {str(row.get("model_id", "")).strip() for row in new_rows}

    # При force полностью заменяем строки только для затронутых моделей.
    if force:
        preserved_rows = [
            row for row in existing_rows if str(row.get("model_id", "")).strip() not in target_model_ids
        ]
        return preserved_rows + new_rows

    # Без force сохраняем существующие ручные оценки и добавляем только отсутствующие строки.
    existing_keys = {make_manual_row_key(row) for row in existing_rows}
    merged_rows = list(existing_rows)
    for row in new_rows:
        if make_manual_row_key(row) not in existing_keys:
            merged_rows.append(row)
            existing_keys.add(make_manual_row_key(row))

    return merged_rows


def write_json(path: Path, payload: object) -> None:
    """Сохраняет объект в json-файл.

    Parameters:
        path (Path): Путь к json-файлу.
        payload (object): Сохраняемый объект.

    Returns:
        None: Ничего не возвращает.

    Fallbacks:
        Если объект не сериализуется, будет выброшена стандартная ошибка TypeError.
    """

    # Сохраняем json в читаемом виде
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def generate_outputs(models: list[dict], model_filter: str, force: bool) -> None:
    """Генерирует аудио и подготавливает шаблон ручной оценки.

    Parameters:
        models (list[dict]): Список конфигураций моделей.
        model_filter (str): Необязательный фильтр по id модели.
        force (bool): Нужно ли перезаписывать wav.

    Returns:
        None: Ничего не возвращает.

    Fallbacks:
        Если фильтр не совпал ни с одной моделью, выбрасывается ошибка.
    """

    # Фильтруем список моделей при отладке
    active_models = models
    if model_filter:
        active_models = [model for model in models if str(model.get("id", "")) == model_filter]
        if not active_models:
            raise ValueError(f"Модель с id={model_filter} не найдена в {MODELS_PATH}")

    # Строим плоский список текстов и кэш моделей
    text_rows = normalize_text_groups()
    run_rows = []
    model_cache = {}

    # Генерируем все пары модель + текст
    for model in active_models:
        for text_row in text_rows:
            run_rows.append(synthesize_item(model, text_row, force, model_cache))

    # Сохраняем автоматический журнал запусков
    write_json(RUNS_PATH, run_rows)

    # Обновляем шаблон ручных оценок, не затрагивая другие модели
    manual_rows = build_manual_rows(run_rows)
    existing_manual_rows = []
    if MANUAL_SCORES_PATH.exists():
        existing_manual_rows = read_manual_rows(MANUAL_SCORES_PATH)
    merged_manual_rows = merge_manual_rows(existing_manual_rows, manual_rows, force)
    write_csv(MANUAL_SCORES_PATH, MANUAL_SCORE_FIELDS, merged_manual_rows)

    print(f"[saved] runs={RUNS_PATH}")
    print(f"[saved] manual_scores={MANUAL_SCORES_PATH}")


def read_manual_rows(path: Path) -> list[dict]:
    """Загружает manual_scores.csv в память.

    Parameters:
        path (Path): Путь к manual_scores.csv.

    Returns:
        list[dict]: Строки csv как словари.

    Fallbacks:
        Если файл отсутствует, выбрасывается FileNotFoundError.
    """

    # Проверяем наличие csv с ручной оценкой
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл ручной оценки: {path}")

    # Читаем csv в список словарей
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str) -> float | None:
    """Преобразует строку в число с плавающей точкой.

    Parameters:
        value (str): Строка из csv.

    Returns:
        float | None: Число или None.

    Fallbacks:
        Пустые и нечисловые значения превращаются в None.
    """

    # Нормализуем пробелы и десятичную запятую
    normalized = str(value).strip().replace(",", ".")
    if not normalized:
        return None

    # Возвращаем число при валидном формате
    try:
        return float(normalized)
    except ValueError:
        return None


def parse_error_flag(value: str) -> bool | None:
    """Преобразует строковый флаг ошибки в bool.

    Parameters:
        value (str): Значение из csv.

    Returns:
        bool | None: Истина, ложь или None.

    Fallbacks:
        Неизвестные значения возвращают None.
    """

    # Нормализуем строку
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized in {"1", "true", "yes", "y", "да", "ошибка"}:
        return True
    if normalized in {"0", "false", "no", "n", "нет", "ok"}:
        return False
    return None


def compute_average_latency(rows: list[dict]) -> float | None:
    """Считает среднее latency по строкам модели.

    Parameters:
        rows (list[dict]): Строки одной модели.

    Returns:
        float | None: Среднее значение или None.

    Fallbacks:
        Если валидных значений нет, возвращается None.
    """

    # Отбираем только заполненные latency
    values = []
    for row in rows:
        latency = parse_float(str(row.get("latency_sec", "")))
        if latency is not None:
            values.append(latency)

    # Возвращаем среднее только при наличии данных
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def compute_group_error_rate(rows: list[dict], group: str) -> float | None:
    """Считает процент ошибок внутри одной группы.

    Parameters:
        rows (list[dict]): Строки одной модели.
        group (str): Название тестовой группы.

    Returns:
        float | None: Процент ошибочных чтений или None.

    Fallbacks:
        Если группа не размечена, возвращается None.
    """

    # Отбираем только размеченные элементы группы
    marked = []
    for row in rows:
        if str(row.get("group", "")) != group:
            continue
        error_flag = parse_error_flag(str(row.get("has_error", "")))
        if error_flag is None:
            continue
        marked.append(error_flag)

    # Возвращаем процент только при наличии разметки
    if not marked:
        return None
    errors = sum(1 for value in marked if value)
    return round((errors / len(marked)) * 100.0, 2)


def collect_stress_notes(rows: list[dict]) -> str:
    """Собирает текстовые заметки по ударению для модели.

    Parameters:
        rows (list[dict]): Строки одной модели.

    Returns:
        str: Объединённые заметки по stress-группе.

    Fallbacks:
        Если заметки не заполнены, возвращается пустая строка.
    """

    # Собираем только непустые заметки по ударению
    notes = []
    for row in rows:
        if str(row.get("group", "")) != "stress":
            continue
        value = str(row.get("stress_notes", "")).strip()
        if value:
            notes.append(f"{row.get('item_id', '')}: {value}")

    return " | ".join(notes)


def summarize_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Готовит плоскую и расширенную сводку по моделям.

    Parameters:
        rows (list[dict]): Строки manual_scores.csv.

    Returns:
        tuple[list[dict], list[dict]]: Короткая и расширенная сводка.

    Fallbacks:
        Если данных нет, возвращаются пустые списки.
    """

    # Группируем строки по модели
    by_model = {}
    for row in rows:
        model_id = str(row.get("model_id", "")).strip()
        if not model_id:
            continue
        by_model.setdefault(model_id, []).append(row)

    # Считаем агрегаты для каждой модели
    summary_rows = []
    summary_json_rows = []
    for model_id, model_rows in sorted(by_model.items()):
        average_latency = compute_average_latency(model_rows)
        graphic_rate = compute_group_error_rate(model_rows, "abbr_graphic")
        lexical_rate = compute_group_error_rate(model_rows, "abbr_lexical")
        numbers_rate = compute_group_error_rate(model_rows, "numbers")
        stress_notes = collect_stress_notes(model_rows)

        summary_rows.append(
            {
                "model_id": model_id,
                "avg_latency_sec": "" if average_latency is None else average_latency,
                "abbr_graphic_error_percent": "" if graphic_rate is None else graphic_rate,
                "abbr_lexical_error_percent": "" if lexical_rate is None else lexical_rate,
                "numbers_error_percent": "" if numbers_rate is None else numbers_rate,
                "stress_notes": stress_notes,
            }
        )

        summary_json_rows.append(
            {
                "model_id": model_id,
                "average_latency_sec": average_latency,
                "error_rates_percent": {
                    "abbr_graphic": graphic_rate,
                    "abbr_lexical": lexical_rate,
                    "numbers": numbers_rate,
                },
                "stress_notes": stress_notes,
                "labels": {
                    "abbr_graphic": GROUP_LABELS["abbr_graphic"],
                    "abbr_lexical": GROUP_LABELS["abbr_lexical"],
                    "numbers": GROUP_LABELS["numbers"],
                    "stress": GROUP_LABELS["stress"],
                },
            }
        )

    return summary_rows, summary_json_rows


def summarize_outputs() -> None:
    """Считывает manual_scores.csv и сохраняет сводные файлы.

    Parameters:
        Нет.

    Returns:
        None: Ничего не возвращает.

    Fallbacks:
        Если manual_scores.csv ещё не создан, выбрасывается FileNotFoundError.
    """

    # Загружаем ручные оценки
    manual_rows = read_manual_rows(MANUAL_SCORES_PATH)

    # Считаем агрегаты по моделям
    summary_rows, summary_json_rows = summarize_rows(manual_rows)

    # Сохраняем csv-таблицу для отчёта
    write_csv(
        SUMMARY_CSV_PATH,
        [
            "model_id",
            "avg_latency_sec",
            "abbr_graphic_error_percent",
            "abbr_lexical_error_percent",
            "numbers_error_percent",
            "stress_notes",
        ],
        summary_rows,
    )

    # Сохраняем json-версию для дальнейшей обработки
    write_json(SUMMARY_JSON_PATH, summary_json_rows)

    print(f"[saved] summary_csv={SUMMARY_CSV_PATH}")
    print(f"[saved] summary_json={SUMMARY_JSON_PATH}")


def main() -> None:
    """Запускает нужный режим работы скрипта.

    Parameters:
        Нет.

    Returns:
        None: Ничего не возвращает.

    Fallbacks:
        При ошибке печатает сообщение в stderr и завершает программу кодом 1.
    """

    # Создаём директории и читаем аргументы
    ensure_directories()
    args = parse_args()

    # Выполняем только запрошенные действия
    try:
        if args.generate:
            models = load_models(MODELS_PATH)
            generate_outputs(models, args.model, args.force)
        if args.summarize:
            summarize_outputs()
    except Exception as error:
        print(f"[fatal] {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
