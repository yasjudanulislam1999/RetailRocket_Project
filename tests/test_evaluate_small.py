from src.evaluate import hit_at_k


def test_hit_at_k():
    recs = [5, 6, 7, 8]
    targets = [100, 7]

    # top2 = [5,6] -> 7 not included
    assert hit_at_k(recs, targets, 2) == 0

    # top3 = [5,6,7] -> 7 included
    assert hit_at_k(recs, targets, 3) == 1
