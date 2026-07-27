from typing import Any

import pandas as pd


def print_audit_summary(report: dict[str, Any]) -> None:
    print("DATASET AUDIT")
    print("-" * 40)
    print("Total images:", report["total_images"])
    print("Positive images:", report["positive_images"])
    print("Background images:", report["background_images"])
    print("Readable images:", report["readable_images"])
    print("Total objects:", report["total_objects"])
    print("Issues:", len(report["issues"]))
    print("Duplicate groups:", len(report["duplicate_groups"]))

    print("\nClass counts:")
    for class_id, count in report["class_counts"].items():
        print(f"  {class_id}: {count}")

    print("\nObject sizes:")
    for size, count in report["size_counts"].items():
        print(f"  {size}: {count}")


def boxes_dataframe(
    report: dict[str, Any],
) -> pd.DataFrame:
    return pd.DataFrame(report["box_records"])


def issues_dataframe(
    report: dict[str, Any],
) -> pd.DataFrame:
    return pd.DataFrame(report["issues"])