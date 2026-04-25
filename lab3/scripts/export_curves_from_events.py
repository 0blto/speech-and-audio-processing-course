import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator


def _load_scalars(event_dir: Path) -> dict[str, list[tuple[int, float]]]:
    if not event_dir.is_dir():
        raise SystemExit(f"Нет каталога: {event_dir}")
    out: dict[str, list[tuple[int, float]]] = defaultdict(list)
    to_try = [event_dir, *[p for p in event_dir.rglob("*") if p.is_dir()]]
    seen: set[Path] = set()
    for d in to_try:
        for evf in d.glob("events.out.tfevents*"):
            p = evf.parent.resolve()
            if p in seen:
                continue
            seen.add(p)
            try:
                ea = event_accumulator.EventAccumulator(
                    str(p),
                    size_guidance={event_accumulator.SCALARS: 0},
                )
                ea.Reload()
            except Exception as e:
                print("Пропуск", evf, e, file=sys.stderr)
                continue
            for tag in ea.Tags().get("scalars", []):
                for s in ea.Scalars(tag):
                    out[tag].append((s.step, s.value))
    for tag, pts in out.items():
        out[tag] = sorted(pts, key=lambda t: t[0])
    return dict(out)


def _plot(cols: dict[str, list[tuple[int, float]]], out_path: Path) -> None:
    if not cols:
        raise SystemExit("Скаляры в events не найдены — дождитесь print_step/логгера в TensorBoard.")
    n = min(len(cols), 6)
    keys = list(cols.keys())[:n]
    fig, axs = plt.subplots(n, 1, figsize=(9, 2.5 * n), squeeze=False)
    for i, k in enumerate(keys):
        ck = cols[k]
        if ck:
            x, y = zip(*ck)
            x, y = list(x), list(y)
        else:
            x, y = [], []
        axs[i, 0].plot(x, y, lw=0.8)
        axs[i, 0].set_title(k)
        axs[i, 0].set_xlabel("step")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    (out_path.parent / "scalar_tags.txt").write_text("\n".join(cols.keys()) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Сохранить кривые обучения из TensorBoard (events) в training_curves.png"
    )
    ap.add_argument("run_dir", type=Path, help="Каталог runs/.../ с events.out.tfevents*")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Путь к PNG (по умолчаню run_dir/figures/training_curves.png)",
    )
    args = ap.parse_args()
    r = args.run_dir.resolve()
    out = (args.out or (r / "figures" / "training_curves.png")).resolve()
    sc = _load_scalars(r)
    _plot(sc, out)
    print("OK:", out, "| теги:", list(sc)[:8], "…" if len(sc) > 8 else "")


if __name__ == "__main__":
    main()
