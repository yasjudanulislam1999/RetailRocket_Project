from collections import Counter
from src.cooccur import _flush_group


def test_flush_group_counts_items_and_pairs():
    item_freq = Counter()
    pair_counts = Counter()

    # One basket with 3 items
    items = {10, 20, 30}
    _flush_group(items, item_freq, pair_counts)

    # Each item appears once
    assert item_freq[10] == 1
    assert item_freq[20] == 1
    assert item_freq[30] == 1

    # Pairs: (10,20), (10,30), (20,30)
    assert pair_counts[(10, 20)] == 1
    assert pair_counts[(10, 30)] == 1
    assert pair_counts[(20, 30)] == 1
