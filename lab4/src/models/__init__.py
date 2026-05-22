"""Экспортирует инструменты реестра моделей.

Fallbacks:
    Если конкретный адаптер недоступен, ошибка произойдёт при запросе адаптера через registry.
"""

# Экспортируем публичные функции реестра
from models.registry import get_model_adapter
from models.registry import load_model_once

