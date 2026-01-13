from __future__ import annotations

import math


def cosine_style_sim(count_ab: int, freq_a: int, freq_b: int) -> float:
    """
    Normalised co-occurrence (cosine-style).

    sim = count_ab / sqrt(freq_a * freq_b)

    Returns 0.0 if inputs are invalid (e.g., zero freq).
    """
    if count_ab <= 0:
        return 0.0
    if freq_a <= 0 or freq_b <= 0:
        return 0.0
    return float(count_ab) / math.sqrt(float(freq_a) * float(freq_b))


def final_score(view_sim: float, buy_sim: float, w_view: float = 1.0, w_buy: float = 3.0) -> float:
    """
    Combine view and buy similarities into one final score.
    """
    return (w_view * view_sim) + (w_buy * buy_sim)
