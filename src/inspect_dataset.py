
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

def inspect_dataset():
    files = sorted(RAW_DIR.glob("*.csv"))

    if not files:
        print("ERROR: No CSV files found in data/raw/")
        return

    print(f"\nTotal CSV files: {len(files)}")
    print(f"Dataset directory: {RAW_DIR}")

    for file in files:
        print("\n" + "=" * 70)
        print(f"FILE: {file.name}")
        print(f"SIZE: {file.stat().st_size / (1024**2):.2f} MB")

        try:
            df = pd.read_csv(file, nrows=5, encoding="utf-8")
            
            # Normalize column names for inspection
            columns = [str(c).strip() for c in df.columns]

            print(f"TOTAL COLUMNS: {len(columns)}")
            print("\nCOLUMN NAMES:")
            for column in columns:
                print(f"  - {column}")

            label_columns = [
                col for col in df.columns
                if str(col).strip().lower() == "label"
            ]

            if label_columns:
                print("\nSAMPLE LABELS:")
                print(df[label_columns[0]].value_counts())

        except Exception as error:
            print(f"Inspection failed: {error}")

if __name__ == "__main__":
    inspect_dataset()
