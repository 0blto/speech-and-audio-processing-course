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

## 1. Сравнение моделей (инференс, без обучения)

Скачает веса при первом запуске (нужен интернет).

```bash
python scripts/inference_compare.py
```

## 2. Данные LJSpeech (для дообучения)

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

## Оценка качества (для отчёта)

- **Разборчивость (intelligibility)**: MOS, тест с перечислением слов, STOI/PESQ при наличии референса.
- **Качество / естественность**: субъективное прослушивание, MCD между mel при наличии эталона.
- **Скорость**: время синтеза (скрипт `inference_compare.py` печатает секунды).

Краткие ожидания: **VITS** часто быстрее и «цельнее» end-to-end; **Tacotron2** может давать характерный тембр LJSpeech с типичным вокодером.
