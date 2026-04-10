import sys

v = sys.version_info
if v.major != 3 or v.minor < 9 or v.minor > 11:
    print(
        f"Ошибка: обнаружен Python {v.major}.{v.minor}.{v.micro}.\n"
        "Установите Python 3.9, 3.10 или 3.11 и создайте виртуальное окружение.\n"
        "Пример (Windows): py -3.11 -m venv .venv && .venv\\Scripts\\activate"
    )
    sys.exit(1)
print(f"OK: Python {v.major}.{v.minor}.{v.micro}")
