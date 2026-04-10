# Лабораторная работа: нейросетевой синтез речи (TTS)

Проект на базе **Coqui TTS** (готовые модели, без реализации архитектур с нуля).

## Модели

| Тип | Имя в Coqui | Комментарий |
|-----|-------------|-------------|
| Классическая (Tacotron2) | `tts_models/en/ljspeech/tacotron2-DDC` | Tacotron2 + отдельный вокодер в составе пайплайна |
| Современная (VITS) | `tts_models/en/ljspeech/vits` | End-to-end, быстрый синтез |

WaveNet в Coqui для LJSpeech обычно не выставлен отдельной готовой моделью; Tacotron2 здесь соответствует «классическому» ветке задания.

## Требования

- **Python 3.9, 3.10 или 3.11** (пакет `TTS==0.22.0` **не** поддерживает Python 3.12+).
- **Windows / Linux / macOS**. GPU необязателен (на CPU дообучение и синтез будут медленнее).
- Для корректной работы **фонем** (англ. LJSpeech-модели) установите **eSpeak NG**:
  - Windows: [релизы eSpeak NG](https://github.com/espeak-ng/espeak-ng/releases) — добавьте `espeak-ng.exe` в `PATH` или укажите путь в документации Coqui.
  - Linux: `sudo apt install espeak-ng` (или аналог).

Скрипт `scripts/inference_compare.py`, если **не** находит `espeak-ng` / `espeak` в `PATH`, сам подставляет для VITS фонемайзер **gruut** (зависимость Coqui, без отдельной установки). Так синтез запускается «из коробки»; для поведения максимально близкого к исходной сборке модели по-прежнему лучше установить eSpeak NG.

## Установка

Из корня проекта:

```bash
python check_env.py
python -m venv .venv
```

Активация:

- Windows (cmd): `.venv\Scripts\activate`
- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- Linux/macOS: `source .venv/bin/activate`

```bash
pip install -U pip
pip install -r requirements.txt
```

Либо через Conda:

```bash
conda env create -f environment.yml
conda activate lab3-tts
```

## Структура проекта

| Путь | Назначение |
|------|------------|
| `check_env.py` | Проверка версии Python |
| `requirements.txt` / `environment.yml` | Зависимости |
| `data/synthesis_text.txt` | ~200 слов английского (вопросы, восклицания, «:», «—») |
| `scripts/download_ljspeech.py` | Скачивание и распаковка LJSpeech в `data/LJSpeech-1.1` |
| `scripts/prepare_subset.py` | `metadata_subset.csv` для короткого дообучения |
| `scripts/finetune.py` | 1 эпоха дообучения Tacotron2 или VITS, логи в `runs/` |
| `scripts/inference_compare.py` | Синтез тем же текстом, wav + mel PNG, замер времени |
| `runs/` | TensorBoard-логи и чекпоинты дообучения |
| `outputs/compare/` | Результаты сравнительного синтеза |

## 1. Сравнение моделей (инференс, без обучения)

Скачает веса при первом запуске (нужен интернет).

```bash
python scripts/inference_compare.py
```

Результат: `outputs/compare/tacotron2_synthesis.wav`, `vits_synthesis.wav`, соответствующие `*_mel.png`.

Опция GPU:

```bash
python scripts/inference_compare.py --gpu
```

## 2. Данные LJSpeech (для дообучения)

Архив ~2.3 ГБ.

```bash
python scripts/download_ljspeech.py
```

```bash
python scripts/prepare_subset.py --num-samples 500
```

## 3. Короткое дообучение (1 эпоха) и эксперименты A/B

Два набора гиперпараметров:

- **Эксперимент A**: `lr = 1e-4`, `batch_size = 8`
- **Эксперимент B**: `lr = 2.5e-5`, `batch_size = 20`

Примеры:

```bash
python scripts/finetune.py --model tacotron2 --exp A
python scripts/finetune.py --model tacotron2 --exp B
python scripts/finetune.py --model vits --exp A
python scripts/finetune.py --model vits --exp B
```

### Логирование (TensorBoard)

В другом терминале:

```bash
tensorboard --logdir runs
```

В логах: **loss**, **learning rate**, **mel**-спектрограммы (и другие метрики, которые пишет конкретная модель).

Сравните кривые **A vs B** для одной модели: при более высоком `lr` шаги обновления крупнее; при большем `batch` градиенты стабильнее, но шагов за эпоху меньше.

### Синтез после дообучения

Укажите путь к лучшему чекпоинту и `config.json` из каталога `runs/...` (см. документацию Coqui):

```bash
tts --text "Hello world" --model_path path/to/best_model.pth --config_path path/to/config.json --out_path outputs/finetuned.wav
```

(Точное имя файла чекпоинта смотрите в папке эксперимента.)

## Оценка качества (для отчёта)

- **Разборчивость (intelligibility)**: MOS, тест с перечислением слов, STOI/PESQ при наличии референса.
- **Качество / естественность**: субъективное прослушивание, MCD между mel при наличии эталона.
- **Скорость**: время синтеза (скрипт `inference_compare.py` печатает секунды).

Краткие ожидания: **VITS** часто быстрее и «цельнее» end-to-end; **Tacotron2** может давать характерный тембр LJSpeech с типичным вокодером.

---

## Вопросы для отчёта

### 1. Какие дополнительные характеристики можно подать на вход модели для улучшения синтеза?

- **Идентификатор говорящего** (speaker id / эмбеддинг) для мультиспикерных моделей.
- **Просодия**: F0 (тон), энергия, темп, ударения, паузы.
- **Лингвистика**: фонемная разметка, ударения, варианты произношения.
- **Стиль / эмоция**: метки стиля, GST, reference-аудио для клонирования тембра.
- **Контекст**: соседние предложения для более связной интонации (в продвинутых системах).

### 2. Чем отличается дообучение от обучения с нуля?

- **Обучение с нуля**: случайная инициализация, нужны большие данные и время, модель учит общие закономерности «с нуля».
- **Дообучение (fine-tuning)**: старт с предобученных весов, меньший `lr`, меньше данных и итераций; цель — адаптировать голос/домен, не «сломав» уже выученные признаки.

---

## Устранение неполадок

- `No matching distribution for TTS`: смените Python на **3.11** и пересоздайте venv.
- Ошибки фонем / `phonemizer`: установите **eSpeak NG** и проверьте `PATH`.
- Долго качается LJSpeech: используйте стабильный интернет; при обрыве удалите неполный `data/LJSpeech-1.1.tar.bz2` и запустите скрипт снова.
