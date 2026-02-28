from __future__ import annotations

import os

from . import evaluate, plot_curves, train
from .presets import PRESETS, get_config


def main() -> None:
    config, preset_name = get_config()
    print("preset:", preset_name)
    print("pairing_protocol:", config.pairing_mode)
    print("available_presets:", ", ".join(sorted(PRESETS)))
    if "EXPERIMENT_PRESET" not in os.environ:
        print("hint: set EXPERIMENT_PRESET to switch configs")
    print("stage: train")
    train.main()
    print("stage: evaluate")
    evaluate.main()
    print("stage: plot_curves")
    plot_curves.main()
    print("status: completed")


if __name__ == "__main__":
    main()
