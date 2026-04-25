# Лабораторная работа: нейросетевой синтез речи (TTS)

Проект на базе **Coqui TTS**: готовые модели, дообучение **VITS** на диктора из **VCTK** (английский мультидикторный корпус). **Tacotron2** здесь **не** дообучается; для сравнения исходных весов без обучения остаётся `scripts/inference_compare.py` (Tacotron2 + VITS из коробки).

## Модели

| Назначение | Coqui-имя | Комментарий |
|------------|-------------|-------------|
| База для fine-tune | `tts_models/en/ljspeech/vits` | VITS, предобучение на LJSpeech (дальше — адаптация к новому диктору из VCTK) |
| Сравнение «без обучения» | `.../tacotron2-DDC` и `.../vits` | Скрипт `inference_compare.py` |

**Идея fine-tune:** тот же английский фонетический/просодический prior в предобученных весах; на данных одного VCTK-диктора меняется акустика (тембр) при сохранении переносимой интонации и беглости.

## Требования

- **Python 3.9, 3.10 или 3.11** (`TTS==0.22.0` не поддерживает 3.12+).
- **Windows / Linux / macOS**. **GPU 3090 (24 ГБ)**: в скрипте включён mixed precision, батч до 20 — обычно укладывается в VRAM; при OOM уменьшите батч в коде/ветке эксперимента.
- **eSpeak NG** в `PATH` — для **фонем** как в исходной LJSpeech-VITS. Без eSpeak Coqui при инференсе может сработать в режиме графем, но **обучение** с фонемами предпочтительнее. Windows: [релизы eSpeak NG](https://github.com/espeak-ng/espeak-ng/releases).

## Установка

```bash
python check_env.py
python -m venv .venv
```

```bash
pip install -U pip
pip install -r requirements.txt
```

Или: `conda env create -f environment.yml` / `conda activate lab3-tts`.

## 1. Сравнение готовых моделей (без обучения)

```bash
python scripts/inference_compare.py
# опционально: --gpu
```

Сохраняются `wav` и **mel**-картинки. Дообученный VITS смотрите в разделе 4.

## 2. VCTK: скачивание и один «низоватый» мужской диктор (англ.)

Полный **VCTK-Corpus-0.92** — около **10 GiB** (долгая загрузка). Официальная ссылка [в коде](scripts/download_vctk.py) Coqui; при сбоях сети можно вручную скачать тот же zip в `data/` и распаковать.

```bash
python scripts/download_vctk.py
```

По умолчанию диктор **`p360`** (мужской, англ. с шотландским акцентом; часто используется как низоватый мужской голос). Список ID см. в документации VCTK; можно заменить, например:

```bash
python scripts/prepare_vctk_speaker.py --vctk-root "ПУТЬ/К/VCTK-Corpus-0.92" --speaker p226
```

Скрипт копирует только `txt/<id>/` и `wav48_silence_trimmed/<id>/` в `data/VCTK_<id>/`.

## 3. Два эксперимента VITS (A и B) — гиперпараметры и бюджет по времени

| Эксперимент | `lr_gen` = `lr_disc` | `batch` | Смысл |
|-------------|----------------------|---------|--------|
| **A** | 1.5e-4 | 8 | выше шаг, мельче батч |
| **B** | 4e-5 | 20 | осторожнее, крупный батч (меньше шагов на эпоху) |

По умолчанию **1800 эпох** (оба). На RTX 3090 при типичных **~0,15–0,35 с на шаг** (зависит от длины сегментов и I/O) обычно укладываются в **~6–12 ч** на **один** ран, т.е. **A + B** часто **не дольше ~20 ч**; если сессии длиннее — уменьшите `--epochs` (одинаково для A и B, чтобы сравнение оставалось честным по объёму эпох).

**Артефакты в `runs/<имя_рана>/`:**

- `config.json` — конфиг, нужный для инференса
- `run_manifest.json` — гиперпараметры и путь к данным
- `*.pth` / `best_model.pth` (если включено сохранение) — веса
- TensorBoard: `events.out.tfevents*`
- `phoneme_cache/`

**Кривая обучения (PNG):** после/во время тренинга

```bash
python scripts/export_curves_from_events.py runs/vits_vctk_finetune_...
```

Пишет `figures/training_curves.png` и `scalar_tags.txt` в этой папке.

**TensorBoard:**

```bash
tensorboard --logdir runs
```

## 4. Синтез с дообученной VITS: аудио + мел + линейный спектр (STFT)

```bash
python scripts/synthesize_finetuned_vits.py --run-dir runs/ИМЯ_РАНА --gpu
# текст: data/synthesis_text.txt  |  выход: outputs/finetuned_vits/
```

Сохраняется: `*.wav`, `*_mel.png`, `*_linear_stft.png`.

## 5. Оценка (для отчёта)

- Разборчивость, естественность (MOS, прослушивание), по желанию STOI/MCD при референсе
- Сравнение **A vs B** по лоссам/мел-логов и по субъективной окраске голоса

## LJSpeech (старый вариант)

- `scripts/prepare_subset.py` — подмножество LJSpeech; **основной сценарий лабы — VCTK и `prepare_vctk_speaker.py`**.
