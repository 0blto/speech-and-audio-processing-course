# Лабораторная работа №4

## Анализ систем синтеза речи

Цель работы: сравнить 5 локально запускаемых open-source систем синтеза русской речи по скорости старта синтеза, чтению сокращений, чтению цифровых обозначений и корректности ударения.

В этой реализации лабораторная организована как полуручной эксперимент:
- генерация аудио, шаблон таблиц и сводные проценты считаются скриптом
- latency до первого звука, ошибки чтения и наблюдения по ударению фиксируются вручную после прослушивания

## Выбранные 5 моделей

1. `Chatterbox Multilingual`
2. `F5-TTS_RUSSIAN`
3. `ESpeech-TTS RL-V2`
4. `Silero TTS V5`
5. `Silero TTS V4`

## Структура проекта

- [src/constants.py](/C:/Users/krish/VSProjects/speech-and-audio-processing-course/lab4/src/constants.py) содержит все тестовые предложения из `TASK.md`
- [src/run_experiment.py](/C:/Users/krish/VSProjects/speech-and-audio-processing-course/lab4/src/run_experiment.py) запускает эксперимент и строит сводку
- [src/models/registry.py](/C:/Users/krish/VSProjects/speech-and-audio-processing-course/lab4/src/models/registry.py) выбирает адаптер модели
- [src/models/chatterbox_model.py](/C:/Users/krish/VSProjects/speech-and-audio-processing-course/lab4/src/models/chatterbox_model.py) ручка для Chatterbox
- [src/models/f5_tts_model.py](/C:/Users/krish/VSProjects/speech-and-audio-processing-course/lab4/src/models/f5_tts_model.py) ручка для F5-TTS_RUSSIAN
- [src/models/espeech_model.py](/C:/Users/krish/VSProjects/speech-and-audio-processing-course/lab4/src/models/espeech_model.py) ручка для ESpeech-TTS
- [src/models/silero_v5_model.py](/C:/Users/krish/VSProjects/speech-and-audio-processing-course/lab4/src/models/silero_v5_model.py) ручка для Silero TTS V5
- [src/models/silero_v4_model.py](/C:/Users/krish/VSProjects/speech-and-audio-processing-course/lab4/src/models/silero_v4_model.py) ручка для Silero TTS V4
- [data/models.json](/C:/Users/krish/VSProjects/speech-and-audio-processing-course/lab4/data/models.json) хранит конфигурации моделей
- `results/audio/` содержит сгенерированные wav
- `results/manual_scores.csv` содержит ручную разметку
- `results/summary.csv` и `results/summary.json` содержат итоговую агрегацию

## Стандартизированный интерфейс моделей

Каждая модель имеет отдельный адаптер в `lab4/src/models` и одинаковый набор функций:
- `validate_config(config)`
- `load_model(config)`
- `prepare_text(config, text)`
- `synthesize(model_handle, config, text, output_path)`

Это позволяет вызывать любую модель из одного места через `registry.py`, не дублируя модель-специфичную логику в `run_experiment.py`.

## Особенности выбранных моделей

### Chatterbox Multilingual

- работает как обычный multilingual TTS
- для русского используется `language_id="ru"`
- по умолчанию запускается без reference audio
- по данным официального репозитория предпочтителен Python `3.11`

### F5-TTS_RUSSIAN

- reference-based модель
- требует `ref_audio_path` и `ref_text`
- поддерживает ручную разметку ударения через символ `+`
- в проекте конфиг настраивается через `models.json`

### ESpeech-TTS RL-V2

- reference-based модель
- использует F5-совместимый inference pipeline
- может использовать `RUAccent` для автоматической расстановки ударений
- также требует `ref_audio_path` и `ref_text`

### Silero TTS V5

- обычный TTS без reference audio
- в конфиге фиксируется одна конкретная русская модель и один конкретный speaker

### Silero TTS V4

- обычный TTS без reference audio
- используется вместо `Silero TTS V3`
- в конфиге фиксируется один speaker

## Как запускать

1. Подготовить `lab4/data/models.json`
2. Скачать нужные веса в `lab4/models/`
3. Положить reference wav-файлы в `lab4/reference/` для `F5-TTS_RUSSIAN` и `ESpeech-TTS`
4. Установить зависимости для конкретных моделей

Базовый запуск генерации:

```bash
python lab4/src/run_experiment.py --generate
```

Запуск одной модели:

```bash
python lab4/src/run_experiment.py --generate --model silero_tts_v5_ru
```

Подсчёт сводки:

```bash
python lab4/src/run_experiment.py --summarize
```

## Методика эксперимента

### 1. Скорость обработки текста

Для каждой модели замеряется время от нажатия воспроизведения до произнесения первого звука. Значение вручную заносится в колонку `latency_sec` файла `results/manual_scores.csv`.

Итоговая метрика:
- среднее значение `latency_sec` по всем строкам данной модели

### 2. Чтение графических сокращений

Используются 10 предложений из приложения А. Для каждого предложения вручную выставляется:
- `has_error = да`, если сокращение прочитано неверно
- `has_error = нет`, если ошибка отсутствует
- `error_notes`, если нужно зафиксировать характер ошибки

Итоговая метрика:
- процент строк с ошибкой в группе `abbr_graphic`

### 3. Чтение аббревиатур

Используются 10 предложений из приложения Б. Разметка ведётся по тем же полям, что и в предыдущем пункте.

Итоговая метрика:
- процент строк с ошибкой в группе `abbr_lexical`

### 4. Чтение цифровых обозначений

Используются 10 предложений из приложения В. Разметка ведётся по полям `has_error` и `error_notes`.

Итоговая метрика:
- процент строк с ошибкой в группе `numbers`

### 5. Ударение

Используется фонетически представительный текст из приложения Г. Для каждой модели наблюдения вручную заносятся в `stress_notes`.

Рекомендуемый формат записи:
- `корректно: ...`
- `сомнительно: ...`
- `ошибка: ...`

Итог:
- качественный вывод по модели без числовой метрики

## Методологическое примечание

Модели используются в двух режимах:
- `Chatterbox`, `Silero TTS V5`, `Silero TTS V4` как обычный TTS без reference audio
- `F5-TTS_RUSSIAN` и `ESpeech-TTS` как reference-based TTS с фиксированным reference clip

Это различие нужно явно указать в отчёте, чтобы сравнение было интерпретируемым.

## Таблицы для отчёта

### Таблица 1. Средняя latency

Заполняется по `results/summary.csv`.

| Модель | Средняя latency, сек |
|---|---|
| `chatterbox_multilingual_ru` | |
| `f5_tts_russian` | |
| `espeech_tts_rl_v2` | |
| `silero_tts_v5_ru` | |
| `silero_tts_v4_ru` | |

Вывод:

### Таблица 2. Ошибки чтения графических сокращений

| Модель | Ошибок, % |
|---|---|
| `chatterbox_multilingual_ru` | |
| `f5_tts_russian` | |
| `espeech_tts_rl_v2` | |
| `silero_tts_v5_ru` | |
| `silero_tts_v4_ru` | |

Вывод:

### Таблица 3. Ошибки чтения аббревиатур

| Модель | Ошибок, % |
|---|---|
| `chatterbox_multilingual_ru` | |
| `f5_tts_russian` | |
| `espeech_tts_rl_v2` | |
| `silero_tts_v5_ru` | |
| `silero_tts_v4_ru` | |

Вывод:

### Таблица 4. Ошибки чтения цифровых обозначений

| Модель | Ошибок, % |
|---|---|
| `chatterbox_multilingual_ru` | |
| `f5_tts_russian` | |
| `espeech_tts_rl_v2` | |
| `silero_tts_v5_ru` | |
| `silero_tts_v4_ru` | |

Вывод:

### Таблица 5. Наблюдения по ударению

| Модель | Наблюдения |
|---|---|
| `chatterbox_multilingual_ru` | |
| `f5_tts_russian` | |
| `espeech_tts_rl_v2` | |
| `silero_tts_v5_ru` | |
| `silero_tts_v4_ru` | |

Вывод:

## Общий вывод

После заполнения таблиц здесь нужно кратко сравнить:
- какая модель быстрее стартует
- какая лучше читает сокращения
- какая лучше читает числа
- у какой модели меньше проблем с ударением
- какая модель в целом показала лучший баланс

## Что автоматизировано

- единый вызов любой модели через стандартизированный адаптер
- генерация wav для всех пар `модель + текст`
- построение `results/manual_scores.csv`
- расчёт средних latency
- расчёт процентов ошибок по 3 числовым критериям
- сбор итоговых `summary.csv` и `summary.json`

## Что оценивается вручную

- latency до первого звука
- наличие ошибки чтения
- характер ошибки
- проблемные места ударения

## Вопросы для самопроверки

### 1. Насколько важна скорость синтеза речи?

Скорость синтеза важна для интерактивных сценариев: голосовых ассистентов, экранных дикторов, навигации и диалоговых систем. Большая задержка перед первым звуком ухудшает пользовательский опыт, даже если итоговое качество голоса высокое.

### 2. Что такое естественность речи?

Естественность речи — это степень близости синтезированной речи к живой человеческой речи по интонации, ритму, ударению, темпу, паузам и отсутствию неестественных артефактов. Чем меньше слушатель замечает искусственное происхождение речи, тем выше её естественность.
