
from pathlib import Path
from collections import Counter
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

CHUNK_SIZE = 100_000


def profile_dataset():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError("No CSV files found in data/raw/")

    total_rows = 0
    overall_labels = Counter()
    file_summaries = []
    reference_columns = None

    print("=" * 75)
    print("CIC-IDS2017 FULL DATASET PROFILING")
    print("=" * 75)

    for file in files:

        print(f"\nProcessing: {file.name}")

        file_rows = 0
        file_labels = Counter()

        try:
            for chunk in pd.read_csv(
                file,
                chunksize=CHUNK_SIZE,
                encoding="utf-8-sig",
                skipinitialspace=True
            ):

                # Normalize column names
                chunk.columns = chunk.columns.astype(str).str.strip()

                # Validate schema
                if "Label" not in chunk.columns:
                    raise ValueError("Label column not found!")

                current_columns = list(chunk.columns)

                if reference_columns is None:
                    reference_columns = current_columns

                elif set(current_columns) != set(reference_columns):
                    raise ValueError(
                        f"Schema mismatch detected in {file.name}"
                    )

                # Extract and normalize labels
                labels = (
                    chunk["Label"]
                    .astype("string")
                    .str.strip()
                )

                labels = labels.fillna("<MISSING>")

                counts = labels.value_counts()

                file_labels.update(counts.to_dict())

                file_rows += len(chunk)

            total_rows += file_rows
            overall_labels.update(file_labels)

            file_summaries.append({
                "file": file.name,
                "rows": file_rows,
                "labels": dict(file_labels)
            })

            print(f"Rows processed: {file_rows:,}")
            print(f"Attack categories: {len(file_labels)}")

            for label, count in sorted(file_labels.items()):
                print(f"  {label}: {count:,}")

        except Exception as error:
            print(f"ERROR processing {file.name}: {error}")
            raise

    # Overall statistics
    print("\n" + "=" * 75)
    print("OVERALL DATASET STATISTICS")
    print("=" * 75)

    print(f"Total files: {len(files)}")
    print(f"Total records: {total_rows:,}")
    print(f"Total columns: {len(reference_columns)}")
    print(f"Candidate features: {len(reference_columns) - 1}")

    print("\nOVERALL CLASS DISTRIBUTION")

    for label, count in sorted(overall_labels.items()):
        percentage = (count / total_rows) * 100 if total_rows else 0

        print(
            f"{label:30} "
            f"{count:12,} "
            f"({percentage:.2f}%)"
        )

    # Save reports
    class_report = pd.DataFrame([
        {
            "Label": label,
            "Count": count,
            "Percentage": round(count / total_rows * 100, 3)
        }
        for label, count in overall_labels.items()
    ])

    class_report = class_report.sort_values(
        "Count",
        ascending=False
    )

    class_report.to_csv(
        REPORT_DIR / "class_distribution.csv",
        index=False
    )

    file_report = pd.DataFrame([
        {
            "File": item["file"],
            "Rows": item["rows"],
            "Attack_Categories": len(item["labels"])
        }
        for item in file_summaries
    ])

    file_report.to_csv(
        REPORT_DIR / "file_summary.csv",
        index=False
    )

    print("\nReports saved successfully!")
    print(f"Location: {REPORT_DIR}")


if __name__ == "__main__":
    profile_dataset()
