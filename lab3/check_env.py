import sys

v = sys.version_info
if v.major != 3 or v.minor < 9 or v.minor > 11:
    print(
        f"Ошибка: обнаружен Python {v.major}.{v.minor}.{v.micro}.\n"
        "Требуется Python 3.9-3.11.\n"
    )
    sys.exit(1)
print(f"OK: Python {v.major}.{v.minor}.{v.micro}")
