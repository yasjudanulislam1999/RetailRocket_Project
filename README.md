# RetailRocket Item-to-Item Recommender (MLOps)

Goal: given an `itemid`, return Top-K related items:
- also viewed (session co-occurrence)
- also bought (transaction basket co-occurrence)

We will build the smallest working version first, then add:
MLflow, DagsHub, S3, Docker, Kubernetes, GitHub Actions.

## Quickstart (Milestone 1)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
