from src.rank import cosine_style_sim, final_score


def test_cosine_style_sim_basic():
    # count=10, freq_a=100, freq_b=25
    # sim = 10 / sqrt(2500) = 10 / 50 = 0.2
    sim = cosine_style_sim(10, 100, 25)
    assert abs(sim - 0.2) < 1e-9


def test_final_score_weights():
    # view=0.5, buy=0.5 -> final = 1*0.5 + 3*0.5 = 2.0
    s = final_score(0.5, 0.5, w_view=1.0, w_buy=3.0)
    assert abs(s - 2.0) < 1e-9
