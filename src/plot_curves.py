from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .presets import get_config


def _latest_curve_csv(report_dir: str, suffix: str) -> Path | None:
    candidates = sorted(Path(report_dir).glob(f"*{suffix}"))
    return candidates[-1] if candidates else None


def _load_curve_points(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({key: float(value) for key, value in row.items()})
        return rows


def _plot_roc(points: list[dict[str, float]], title: str, out_path: Path) -> None:
    fars = [point["far"] for point in points]
    tars = [point["tar"] for point in points]

    plt.figure(figsize=(6, 5))
    plt.plot(fars, tars, linewidth=2)
    plt.xlabel("False Accept Rate")
    plt.ylabel("True Accept Rate")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_det(points: list[dict[str, float]], title: str, out_path: Path) -> None:
    fars = [point["far"] for point in points]
    frrs = [point["frr"] for point in points]

    plt.figure(figsize=(6, 5))
    plt.plot(fars, frrs, linewidth=2)
    plt.xlabel("False Accept Rate")
    plt.ylabel("False Reject Rate")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _stem_without_suffix(path: Path, suffix: str) -> str:
    name = path.name
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def main() -> None:
    config, preset_name = get_config()
    global_csv = _latest_curve_csv(config.eval_report_dir, "_global.csv")
    if global_csv is None:
        print("status: missing")
        print("reason: no global curve CSV found in", config.eval_report_dir)
        return

    block_csv = global_csv.with_name(global_csv.name.replace("_global.csv", "_block.csv"))
    if not block_csv.exists():
        print("status: missing")
        print("reason: matching block curve CSV not found for", global_csv.name)
        return

    global_points = _load_curve_points(global_csv)
    block_points = _load_curve_points(block_csv)

    stem = _stem_without_suffix(global_csv, "_global.csv")
    global_roc = global_csv.with_name(f"{stem}_global_roc.png")
    global_det = global_csv.with_name(f"{stem}_global_det.png")
    block_roc = block_csv.with_name(f"{stem}_block_roc.png")
    block_det = block_csv.with_name(f"{stem}_block_det.png")

    _plot_roc(global_points, f"ROC (Global) - {stem}", global_roc)
    _plot_det(global_points, f"DET (Global) - {stem}", global_det)
    _plot_roc(block_points, f"ROC (Block) - {stem}", block_roc)
    _plot_det(block_points, f"DET (Block) - {stem}", block_det)

    print("preset:", preset_name)
    print("pairing_protocol:", config.pairing_mode)
    print("global_csv:", str(global_csv))
    print("block_csv:", str(block_csv))
    print("global_roc_png:", str(global_roc))
    print("global_det_png:", str(global_det))
    print("block_roc_png:", str(block_roc))
    print("block_det_png:", str(block_det))


if __name__ == "__main__":
    main()
