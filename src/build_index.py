from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from src.rank import cosine_style_sim, final_score
from src.config import Config


def _load_item_freq(path: Path) -> dict[int, int]:
    df = pd.read_csv(path)
    return {int(r.itemid): int(r.freq) for r in df.itertuples(index=False)}


def _load_pair_counts(path: Path) -> pd.DataFrame:
    # columns: item_i,item_j,count
    df = pd.read_csv(path)
    df["item_i"] = df["item_i"].astype("int64")
    df["item_j"] = df["item_j"].astype("int64")
    df["count"] = df["count"].astype("int64")
    return df


def build_topk(
    view_item_freq: dict[int, int],
    view_pairs: pd.DataFrame,
    buy_item_freq: dict[int, int],
    buy_pairs: pd.DataFrame,
    topk: int,
    w_view: float,
    w_buy: float,
) -> dict[int, list[int]]:
    """
    Build final Top-K related items per item.
    Output: { itemid: [related_itemid1, related_itemid2, ...] }
    """
    # We'll store scores in a dict of dicts:
    # scores[a][b] = final_score
    scores: dict[int, dict[int, float]] = {}

    def add_edge(a: int, b: int, score: float) -> None:
        if score <= 0:
            return
        scores.setdefault(a, {})
        # If same pair appears from multiple sources, keep the max (safe + simple)
        prev = scores[a].get(b, 0.0)
        if score > prev:
            scores[a][b] = score

    # ---- 1) View contributions ----
    for r in view_pairs.itertuples(index=False):
        a = int(r.item_i)
        b = int(r.item_j)
        c = int(r.count)

        fa = view_item_freq.get(a, 0)
        fb = view_item_freq.get(b, 0)
        v = cosine_style_sim(c, fa, fb)
        s = final_score(view_sim=v, buy_sim=0.0, w_view=w_view, w_buy=w_buy)

        # store both directions (A->B and B->A)
        add_edge(a, b, s)
        add_edge(b, a, s)

    # ---- 2) Buy contributions ----
    for r in buy_pairs.itertuples(index=False):
        a = int(r.item_i)
        b = int(r.item_j)
        c = int(r.count)

        fa = buy_item_freq.get(a, 0)
        fb = buy_item_freq.get(b, 0)
        bsim = cosine_style_sim(c, fa, fb)
        s = final_score(view_sim=0.0, buy_sim=bsim, w_view=w_view, w_buy=w_buy)

        add_edge(a, b, s)
        add_edge(b, a, s)

    # ---- 3) Convert scores to Top-K list ----
    topk_map: dict[int, list[int]] = {}

    for item, neigh in scores.items():
        # neigh is {other_item: score}
        # sort by score desc, take topk
        sorted_items = sorted(neigh.items(), key=lambda x: x[1], reverse=True)[:topk]
        topk_map[item] = [int(other) for other, _score in sorted_items]

    return topk_map


def main() -> None:
    cfg = Config()  # reads TOPK, VIEW_WEIGHT, BUY_WEIGHT from env or defaults

    artefacts = Path("artefacts")
    out_dir = Path("artefacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    view_item_freq_path = artefacts / "view_item_freq.csv"
    view_pairs_path = artefacts / "view_pair_counts.csv"
    buy_item_freq_path = artefacts / "buy_item_freq.csv"
    buy_pairs_path = artefacts / "buy_pair_counts.csv"

    for p in [view_item_freq_path, view_pairs_path, buy_item_freq_path, buy_pairs_path]:
        if not p.exists():
            print(f"ERROR: missing {p}")
            print("Fix: run python src/cooccur.py first")
            return

    print("Loading artefacts...")
    view_item_freq = _load_item_freq(view_item_freq_path)
    buy_item_freq = _load_item_freq(buy_item_freq_path)

    view_pairs = _load_pair_counts(view_pairs_path)
    buy_pairs = _load_pair_counts(buy_pairs_path)

    print("Building Top-K mapping...")
    topk_map = build_topk(
        view_item_freq=view_item_freq,
        view_pairs=view_pairs,
        buy_item_freq=buy_item_freq,
        buy_pairs=buy_pairs,
        topk=cfg.topk,
        w_view=cfg.view_weight,
        w_buy=cfg.buy_weight,
    )

    # Save as JSON (easy to load in API)
    out_json = out_dir / "topk.json"
    with open(out_json, "w") as f:
        # JSON keys must be strings, so we convert int keys to str
        json.dump({str(k): v for k, v in topk_map.items()}, f)

    # Save a small preview CSV (first 200 items) for humans
    rows = []
    for i, (k, recs) in enumerate(topk_map.items()):
        if i >= 200:
            break
        rows.append({"itemid": k, "top_items": " ".join(str(x) for x in recs[:10])})
    pd.DataFrame(rows).to_csv(out_dir / "topk_preview.csv", index=False)

    print("DONE ✅")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_dir / 'topk_preview.csv'}")
    print(f"Items with recommendations: {len(topk_map)}")


if __name__ == "__main__":
    main()
