from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import mlflow

from src.config import Config
from dotenv import load_dotenv
load_dotenv()


def load_topk(path: Path) -> Dict[int, List[int]]:
    """
    Load topk.json which has string keys in JSON.
    Convert keys to int for easier use.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return {int(k): [int(x) for x in v] for k, v in data.items()}


def hit_at_k(recs: List[int], targets: List[int], k: int) -> int:
    """
    Returns 1 if ANY target is in the top-k recommendations, else 0.
    """
    if not recs or not targets:
        return 0
    topk = set(recs[:k])
    for t in targets:
        if t in topk:
            return 1
    return 0


def eval_transactions(events_path: Path, topk_map: Dict[int, List[int]]) -> Dict[str, float]:
    """
    Evaluate on purchase baskets (transactionid).
    This dataset is small (~22k purchase rows), so we evaluate all baskets.
    """
    df = pd.read_csv(events_path, usecols=["event", "transactionid", "itemid"])
    df = df[(df["event"] == "transaction") & (df["transactionid"].notna())]
    df["transactionid"] = df["transactionid"].astype("int64")
    df["itemid"] = df["itemid"].astype("int64")

    total = 0
    hits10 = 0
    hits20 = 0
    hits50 = 0
    skipped_small = 0
    skipped_no_recs = 0

    for _tid, g in df.groupby("transactionid"):
        items = sorted(set(int(x) for x in g["itemid"].unique()))
        if len(items) < 2:
            skipped_small += 1
            continue

        # simplest query choice: first item
        query = items[0]
        targets = items[1:]

        recs = topk_map.get(query)
        if not recs:
            skipped_no_recs += 1
            continue

        total += 1
        hits10 += hit_at_k(recs, targets, 10)
        hits20 += hit_at_k(recs, targets, 20)
        hits50 += hit_at_k(recs, targets, 50)

    def safe_rate(x: int, d: int) -> float:
        return float(x) / float(d) if d > 0 else 0.0

    return {
        "buy_total_tested": total,
        "buy_skipped_small": skipped_small,
        "buy_skipped_no_recs": skipped_no_recs,
        "buy_hit@10": safe_rate(hits10, total),
        "buy_hit@20": safe_rate(hits20, total),
        "buy_hit@50": safe_rate(hits50, total),
    }


def eval_sessions_stream(
    sessions_path: Path,
    topk_map: Dict[int, List[int]],
    max_sessions: int,
    chunksize: int = 200_000,
) -> Dict[str, float]:
    """
    Evaluate on view sessions (session_id), streaming the file.
    We only evaluate the FIRST max_sessions sessions to keep it fast.
    """
    total = 0
    hits10 = 0
    hits20 = 0
    hits50 = 0
    skipped_small = 0
    skipped_no_recs = 0

    current_session = None
    items_in_session = set()

    cols = ["session_id", "event", "itemid"]

    def flush_one_session(items: set) -> Tuple[int, int, int, int, int, int]:
        nonlocal total, hits10, hits20, hits50, skipped_small, skipped_no_recs

        items_sorted = sorted(int(x) for x in items)
        if len(items_sorted) < 2:
            skipped_small += 1
            return (0, 0, 0, 0, 0, 0)

        query = items_sorted[0]
        targets = items_sorted[1:]

        recs = topk_map.get(query)
        if not recs:
            skipped_no_recs += 1
            return (0, 0, 0, 0, 0, 0)

        total += 1
        hits10 += hit_at_k(recs, targets, 10)
        hits20 += hit_at_k(recs, targets, 20)
        hits50 += hit_at_k(recs, targets, 50)
        return (1, 0, 0, 0, 0, 0)

    sessions_seen = 0

    for chunk in pd.read_csv(sessions_path, usecols=cols, chunksize=chunksize):
        # keep only views
        chunk = chunk[chunk["event"] == "view"]

        for row in chunk.itertuples(index=False):
            sid = row.session_id
            itemid = int(row.itemid)

            if current_session is None:
                current_session = sid

            if sid != current_session:
                # finish previous session
                flush_one_session(items_in_session)
                sessions_seen += 1
                if sessions_seen >= max_sessions:
                    break
                items_in_session = set()
                current_session = sid

            items_in_session.add(itemid)

        if sessions_seen >= max_sessions:
            break

    # flush last session (if we stopped naturally)
    if sessions_seen < max_sessions and items_in_session:
        flush_one_session(items_in_session)
        sessions_seen += 1

    def safe_rate(x: int, d: int) -> float:
        return float(x) / float(d) if d > 0 else 0.0

    return {
        "view_sessions_seen": sessions_seen,
        "view_total_tested": total,
        "view_skipped_small": skipped_small,
        "view_skipped_no_recs": skipped_no_recs,
        "view_hit@10": safe_rate(hits10, total),
        "view_hit@20": safe_rate(hits20, total),
        "view_hit@50": safe_rate(hits50, total),
    }


def main() -> None:
    cfg = Config()
    user = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN")

    print("DAGSHUB_USERNAME set:", bool(user))
    print("DAGSHUB_TOKEN set:", bool(token))

    if user and token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    else:
        print("WARNING: Missing DAGSHUB_USERNAME or DAGSHUB_TOKEN. Remote logging will fail.")

    topk_path = Path("artefacts/topk.json")
    sessions_path = Path("data/processed/sessions.csv")
    events_path = Path("data/raw/events.csv")
    out_report = Path("artefacts/eval_report.json")
    preview_path = Path("artefacts/topk_preview.csv")

    if not topk_path.exists():
        print(f"ERROR: missing {topk_path}")
        print("Fix: run python -m src.build_index first")
        return

    if not sessions_path.exists():
        print(f"ERROR: missing {sessions_path}")
        print("Fix: run python src/sessions.py first")
        return

    if not events_path.exists():
        print(f"ERROR: missing {events_path}")
        return

    # How many sessions to evaluate (keep fast)
    max_sessions = int(os.getenv("EVAL_MAX_SESSIONS", "50000"))

    print("Loading topk.json...")
    topk_map = load_topk(topk_path)

    print("Evaluating buy baskets (transactions)...")
    buy_metrics = eval_transactions(events_path, topk_map)

    print("Evaluating view baskets (sessions) - streaming...")
    view_metrics = eval_sessions_stream(sessions_path, topk_map, max_sessions=max_sessions)

    report = {
        "params": {
            "topk": cfg.topk,
            "view_weight": cfg.view_weight,
            "buy_weight": cfg.buy_weight,
            "eval_max_sessions": max_sessions,
        },
        "metrics": {**buy_metrics, **view_metrics},
    }

    # Save JSON report
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w") as f:
        json.dump(report, f, indent=2)

    # ---- MLflow logging (local) ----

    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    print("MLFLOW_TRACKING_URI =", tracking_uri)  # so we can SEE what is being used
    # Put runs under a meaningful experiment name (creates it if missing)
    mlflow.set_experiment("retailrocket-item2item")

    run_name = f"wV{cfg.view_weight}_wB{cfg.buy_weight}_topk{cfg.topk}_eval{max_sessions}"

    with mlflow.start_run(run_name=run_name):

        # Params
        mlflow.log_param("topk", cfg.topk)
        mlflow.log_param("view_weight", cfg.view_weight)
        mlflow.log_param("buy_weight", cfg.buy_weight)
        mlflow.log_param("eval_max_sessions", max_sessions)

        #
        # BUY
        mlflow.log_metric("buy/hit_10", buy_metrics["buy_hit@10"])
        mlflow.log_metric("buy/hit_20", buy_metrics["buy_hit@20"])
        mlflow.log_metric("buy/hit_50", buy_metrics["buy_hit@50"])

        # VIEW
        mlflow.log_metric("view/hit_10", view_metrics["view_hit@10"])
        mlflow.log_metric("view/hit_20", view_metrics["view_hit@20"])
        mlflow.log_metric("view/hit_50", view_metrics["view_hit@50"])


        # Artefacts
        mlflow.log_artifact(str(out_report))
        if preview_path.exists():
            mlflow.log_artifact(str(preview_path))

    print("\nDONE ✅")
    print(f"Saved eval report: {out_report}")
    print("MLflow saved run to: ",tracking_uri)
    print("\nQuick metrics:")
    print(f"BUY  Hit@10={buy_metrics['buy_hit@10']:.4f}  Hit@20={buy_metrics['buy_hit@20']:.4f}  Hit@50={buy_metrics['buy_hit@50']:.4f}")
    print(f"VIEW Hit@10={view_metrics['view_hit@10']:.4f}  Hit@20={view_metrics['view_hit@20']:.4f}  Hit@50={view_metrics['view_hit@50']:.4f}")


if __name__ == "__main__":
    main()
