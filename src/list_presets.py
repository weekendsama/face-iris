from __future__ import annotations

from dataclasses import asdict

from .presets import PRESET_DEFAULT, PRESETS


def _format_value(value) -> str:
    if isinstance(value, str):
        return value
    return repr(value)


def main() -> None:
    baseline_name = PRESET_DEFAULT
    baseline = PRESETS[baseline_name]
    baseline_dict = asdict(baseline)

    print("default_preset:", baseline_name)
    print("available_presets:", ", ".join(sorted(PRESETS)))

    for name in sorted(PRESETS):
        config = PRESETS[name]
        current = asdict(config)
        print(f"\n[{name}]")
        changed = 0
        for key in sorted(current):
            if current[key] != baseline_dict[key]:
                changed += 1
                print(
                    "{key}: {value} (baseline={baseline})".format(
                        key=key,
                        value=_format_value(current[key]),
                        baseline=_format_value(baseline_dict[key]),
                    )
                )
        if changed == 0:
            print("same as baseline")
        print("pairing_protocol:", current["pairing_mode"])


if __name__ == "__main__":
    main()
