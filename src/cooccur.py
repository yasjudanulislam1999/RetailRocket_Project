from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
import pandas as pd


def _flush_group(
    items_in_group: set[int],
    item_freq: Counter[int],
    pair_counts: Counter[tuple[int, int]],
) -> None:
    """
    Update counters for one group (one session or one transaction).

    - item_freq[item] += 1 for each unique item in the group
    - pair_counts[(a,b)] += 1 for each unique unordered pair in the group
    """
    if not items_in_group:
        return

    # Count item frequency (once per group)
    for it in items_in_group:
        item_freq[it] += 1

    # Count pair co-occurrence (once per group)
    # Sort to keep pairs consistent: (small, big)
    items_sorted = sorted(items_in_group)
    for a, b in combinations(items_sorted, 2):
        pair_counts[(a, b)] += 1


def cooccur_from_sessions_csv(
    sessions_path: Path,
    chunksize: int = 200_000,
) -> tuple[Counter[int], Counter[tuple[int, int]]]:
    """
    Build 'also viewed' co-occurrence from sessions.csv.

    We only use rows where event == 'view'.
    We treat each session_id as one group/basket.

    Important: sessions.csv is already in visitor/time order,
    so rows for the same session_id are contiguous. That lets us stream safely.
    """
    item_freq: Counter[int] = Counter()
    pair_counts: Counter[tuple[int, int]] = Counter()

    current_session: str | None = None
    items_in_session: set[int] = set()

    cols = ["session_id", "event", "itemid"]

    for chunk in pd.read_csv(sessions_path, usecols=cols, chunksize=chunksize):
        # Keep only views (also viewed signal)
        chunk = chunk[chunk["event"] == "view"]

        for row in chunk.itertuples(index=False):
            session_id = row.session_id
            itemid = int(row.itemid)

            # New session begins -> flush previous session
            if current_session is None:
                current_session = session_id

            if session_id != current_session:
                _flush_group(items_in_session, item_freq, pair_counts)
                items_in_session = set()
                current_session = session_id

            items_in_session.add(itemid)

    # Flush the final session
    _flush_group(items_in_session, item_freq, pair_counts)

    return item_freq, pair_counts


def cooccur_from_transactions(
    events_path: Path,
) -> tuple[Counter[int], Counter[tuple[int, int]]]:
    """
    Build 'also bought' co-occurrence from events.csv.

    We use only rows where event == 'transaction' and transactionid exists.
    We group by transactionid (basket).
    """
    df = pd.read_csv(events_path, usecols=["event", "transactionid", "itemid"])

    # Keep only purchase rows
    df = df[(df["event"] == "transaction") & (df["transactionid"].notna())]

    # transactionid sometimes loads as float because of NaNs; convert safely
    df["transactionid"] = df["transactionid"].astype("int64")
    df["itemid"] = df["itemid"].astype("int64")

    item_freq: Counter[int] = Counter()
    pair_counts: Counter[tuple[int, int]] = Counter()

    # Each transaction is one basket
    for tid, g in df.groupby("transactionid"):
        items = set(int(x) for x in g["itemid"].unique())
        _flush_group(items, item_freq, pair_counts)

    return item_freq, pair_counts


def _save_item_freq(path: Path, item_freq: Counter[int]) -> None:
    rows = [{"itemid": k, "freq": v} for k, v in item_freq.items()]
    pd.DataFrame(rows).to_csv(path, index=False)


def _save_pair_counts(path: Path, pair_counts: Counter[tuple[int, int]]) -> None:
    rows = [{"item_i": a, "item_j": b, "count": c} for (a, b), c in pair_counts.items()]
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    sessions_path = Path("data/processed/sessions.csv")
    events_path = Path("data/raw/events.csv")
    out_dir = Path("artefacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not sessions_path.exists():
        print(f"ERROR: missing {sessions_path}")
        print("Fix: run python src/sessions.py first")
        return

    if not events_path.exists():
        print(f"ERROR: missing {events_path}")
        return

    print("Building ALSO VIEWED (session co-occurrence)...")
    view_item_freq, view_pair_counts = cooccur_from_sessions_csv(sessions_path)

    _save_item_freq(out_dir / "view_item_freq.csv", view_item_freq)
    _save_pair_counts(out_dir / "view_pair_counts.csv", view_pair_counts)

    print("✅ Saved artefacts/view_item_freq.csv and artefacts/view_pair_counts.csv")
    print(f"Unique items (view): {len(view_item_freq)}")
    print(f"Unique pairs (view): {len(view_pair_counts)}")

    print("\nBuilding ALSO BOUGHT (transaction co-occurrence)...")
    buy_item_freq, buy_pair_counts = cooccur_from_transactions(events_path)

    _save_item_freq(out_dir / "buy_item_freq.csv", buy_item_freq)
    _save_pair_counts(out_dir / "buy_pair_counts.csv", buy_pair_counts)

    print("✅ Saved artefacts/buy_item_freq.csv and artefacts/buy_pair_counts.csv")
    print(f"Unique items (buy): {len(buy_item_freq)}")
    print(f"Unique pairs (buy): {len(buy_pair_counts)}")

    print("\nDONE ✅")


if __name__ == "__main__":
    main()
