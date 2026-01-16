from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Query


app = FastAPI(title="RetailRocket Item2Item Recommender")

# This will hold: itemid -> list of recommended itemids
TOPK: Dict[int, List[int]] = {}


def load_topk(path: Path) -> Dict[int, List[int]]:
    """
    Load topk.json into a Python dict with int keys.
    JSON keys are strings, so we convert to int.
    """
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): [int(x) for x in v] for k, v in raw.items()}


@app.on_event("startup")
def startup_load_model() -> None:
    """
    When the API starts, load the topk mapping once.
    """
    global TOPK
    path = Path("artefacts/topk.json")
    if not path.exists():
        # Keep it simple: load empty if file missing
        TOPK = {}
        return
    TOPK = load_topk(path)


@app.get("/health")
def health() -> Dict[str, str]:
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}


@app.get("/recommend")
def recommend(
    itemid: int,
    k: int = Query(10, ge=1, le=50),
) -> Dict[str, Optional[object]]:
    """
    Return top-k recommendations for a given itemid.
    """
    recs = TOPK.get(itemid, [])
    return {
        "itemid": itemid,
        "k": k,
        "recommendations": recs[:k],
        "available": itemid in TOPK,
    }
