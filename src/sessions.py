import pandas as pd
from pathlib import Path


def build_sessions(events: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
    """
    Add a session_id to each event based on time gaps per visitor.

    Rules:
    - Sort by visitorid then timestamp
    - New session starts when time gap > gap_minutes
    """
    df = events.copy()

    # Make sure required columns exist
    required = {"visitorid", "timestamp", "event", "itemid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(list(missing))}")

    # Sort so time gaps make sense
    df = df.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)

    # Compute time difference (in minutes) between consecutive events per visitor
    # diff() gives NaN for the first event of each visitor (we treat that as new session)
    df["prev_ts"] = df.groupby("visitorid")["timestamp"].shift(1)
    df["gap_min"] = (df["timestamp"] - df["prev_ts"]) / 1000.0 / 60.0

    # Start a new session if:
    # - this is the first event for the visitor (prev_ts is NaN)
    # - OR the gap is greater than the threshold
    df["new_session"] = df["prev_ts"].isna() | (df["gap_min"] > gap_minutes)

    # Convert True/False to 1/0 then cumulative sum per visitor to get session number
    df["session_num"] = df.groupby("visitorid")["new_session"].cumsum()

    # Build a simple string session_id
    # Example: "12345_1" means visitor 12345 session 1
    df["session_id"] = df["visitorid"].astype(str) + "_" + df["session_num"].astype(int).astype(str)

    # Cleanup helper columns
    df = df.drop(columns=["prev_ts", "gap_min", "new_session", "session_num"])

    return df


def main() -> None:
    raw_path = Path("data/raw/events.csv")
    out_path = Path("data/processed/sessions.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        print(f"ERROR: missing {raw_path}")
        return

    # Load only what we need for sessions (faster)
    events = pd.read_csv(
        raw_path,
        usecols=["timestamp", "visitorid", "event", "itemid", "transactionid"],
    )

    sessions = build_sessions(events, gap_minutes=30)

    # Save processed sessions
    sessions.to_csv(out_path, index=False)

    print("DONE ✅")
    print(f"Saved: {out_path}")
    print(f"Rows: {len(sessions)}")
    print(f"Unique sessions: {sessions['session_id'].nunique()}")

if __name__ == "__main__":
    main()
