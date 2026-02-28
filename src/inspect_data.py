from __future__ import annotations

from .config import DEFAULT_CONFIG
from .data import summarize_dataset_structure, summarize_public_dataset_structure


def main() -> None:
    config = DEFAULT_CONFIG
    if config.use_public_dataset_roots:
        summary = summarize_public_dataset_structure(
            config.lfw_root,
            config.casia_iris_root,
        )
        print("pairing_protocol:", config.pairing_mode)
        print("pairing_mode:", summary["pairing_mode"])
        print("lfw_root:", summary["lfw_root"])
        print("casia_root:", summary["casia_root"])
        print("lfw_identities:", summary["lfw_identities"])
        print("casia_identities:", summary["casia_identities"])
        print("pairable_identities:", summary["pairable_identities"])
        print("total_pairs:", summary["total_pairs"])
        print("note: LFW and CASIA identities are manually aligned by sorted index in this public-data protocol.")
        for item in summary["classes"][:10]:
            print(
                "label={label}: face={face} ({face_count}) iris={iris} ({iris_count}) pairs={pairs}".format(
                    label=item["label"],
                    face=item["face_identity"],
                    face_count=item["face_count"],
                    iris=item["iris_identity"],
                    iris_count=item["iris_count"],
                    pairs=item["pair_count"],
                )
            )
        return

    summary = summarize_dataset_structure(config.dataset_root)

    print("pairing_protocol:", config.pairing_mode)
    print("dataset_root:", summary["dataset_root"])
    if not summary["train_root_exists"]:
        print("status: missing")
        print("expected:", f"{config.dataset_root}/train/<person_id>/face and iris")
        return

    classes = summary["classes"]
    print("status:", "ok" if classes else "empty")
    print("identities:", len(classes))
    print("total_pairs:", summary["total_pairs"])

    for item in classes:
        print(
            "{name}: face_dir={face_dir} iris_dir={iris_dir} face={face} iris={iris} pairs={pairs}".format(
                name=item["name"],
                face_dir="yes" if item["face_dir_exists"] else "no",
                iris_dir="yes" if item["iris_dir_exists"] else "no",
                face=item["face_count"],
                iris=item["iris_count"],
                pairs=item["pair_count"],
            )
        )


if __name__ == "__main__":
    main()
