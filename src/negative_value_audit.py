
from pathlib import Path
from collections import defaultdict

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

CHUNK_SIZE = 100_000

# These two features require separate treatment.
SENTINEL_FEATURES = {
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward"
}


def audit_negative_values():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    negative_stats = defaultdict(lambda: {
        "negative_count": 0,
        "minimum_negative": None
    })

    sentinel_stats = defaultdict(lambda: {
        "minus_one_count": 0,
        "below_minus_one_count": 0
    })

    duplicate_column_mismatches = 0
    comparison_count = 0

    files = sorted(RAW_DIR.glob("*.csv"))

    for file in files:

        print(f"\nProcessing: {file.name}")

        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE,
            encoding="utf-8-sig",
            skipinitialspace=True
        ):

            chunk.columns = chunk.columns.astype(str).str.strip()

            features = chunk.drop(columns=["Label"])

            for column in features.columns:

                values = pd.to_numeric(
                    features[column],
                    errors="coerce"
                )

                if column in SENTINEL_FEATURES:

                    sentinel_stats[column]["minus_one_count"] += int(
                        values.eq(-1).sum()
                    )

                    sentinel_stats[column]["below_minus_one_count"] += int(
                        values.lt(-1).sum()
                    )

                    continue

                negative_mask = values.lt(0)

                count = int(negative_mask.sum())

                if count:

                    minimum = float(values[negative_mask].min())

                    negative_stats[column]["negative_count"] += count

                    previous = negative_stats[column]["minimum_negative"]

                    if previous is None:
                        negative_stats[column]["minimum_negative"] = minimum
                    else:
                        negative_stats[column]["minimum_negative"] = min(
                            previous,
                            minimum
                        )

            # Check whether the two header-length columns
            # are actually identical.
            if (
                "Fwd Header Length" in features.columns
                and "Fwd Header Length.1" in features.columns
            ):

                first = pd.to_numeric(
                    features["Fwd Header Length"],
                    errors="coerce"
                )

                second = pd.to_numeric(
                    features["Fwd Header Length.1"],
                    errors="coerce"
                )

                equal = first.eq(second) | (
                    first.isna() & second.isna()
                )

                duplicate_column_mismatches += int((~equal).sum())

                comparison_count += len(chunk)

        print("Completed:", file.name)

    # Negative feature report
    negative_report = pd.DataFrame([
        {
            "Feature": feature,
            "Negative_Rows": stats["negative_count"],
            "Minimum_Negative_Value": stats["minimum_negative"]
        }
        for feature, stats in negative_stats.items()
    ])

    negative_report = negative_report.sort_values(
        "Negative_Rows",
        ascending=False
    )

    negative_report.to_csv(
        REPORT_DIR / "negative_value_audit.csv",
        index=False
    )

    # Sentinel report
    sentinel_report = pd.DataFrame([
        {
            "Feature": feature,
            "Minus_One_Count": stats["minus_one_count"],
            "Below_Minus_One_Count": stats["below_minus_one_count"]
        }
        for feature, stats in sentinel_stats.items()
    ])

    sentinel_report.to_csv(
        REPORT_DIR / "sentinel_value_audit.csv",
        index=False
    )

    # Duplicate feature comparison
    duplicate_report = pd.DataFrame([{
        "Columns_Compared": "Fwd Header Length / Fwd Header Length.1",
        "Total_Comparisons": comparison_count,
        "Mismatched_Rows": duplicate_column_mismatches
    }])

    duplicate_report.to_csv(
        REPORT_DIR / "duplicate_column_audit.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("NEGATIVE VALUE AUDIT RESULTS")
    print("=" * 65)

    print(negative_report.to_string(index=False))

    print("\nSENTINEL VALUE RESULTS")
    print(sentinel_report.to_string(index=False))

    print("\nDUPLICATE COLUMN COMPARISON")
    print(duplicate_report.to_string(index=False))

    print("\nReports saved successfully!")


if __name__ == "__main__":
    audit_negative_values()
