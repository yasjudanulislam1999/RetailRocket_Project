import pandas as pd
from pathlib import Path


def main() -> None:
    # Where the file should be
    path = Path("data/raw/events.csv")

    # Stop early if file is missing
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        print("Fix: Put events.csv inside data/raw/ so the path is data/raw/events.csv")
        return

    # Read CSV
    df = pd.read_csv(path)

    print("=== BASIC INFO ===")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    print("\n=== NULL COUNTS ===")
    print(df.isna().sum().to_string())

    if "event" in df.columns:
        print("\n=== EVENT COUNTS ===")
        print(df["event"].value_counts().to_string())
    else:
        print("\nWARNING: 'event' column not found")

    if "timestamp" in df.columns:
        print("\n=== TIMESTAMP RANGE (RAW) ===")
        print(f"min: {df['timestamp'].min()}")
        print(f"max: {df['timestamp'].max()}")
    else:
        print("\nWARNING: 'timestamp' column not found")

    print("\nDONE ✅")


if __name__ == "__main__":
    main()
