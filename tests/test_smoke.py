from src.config import Config


def test_smoke_config_defaults():
    """
    Smoke test = tiny test that proves the project runs.
    """
    cfg = Config()

    # Check basic defaults (these should match .env.example defaults)
    assert cfg.topk == 50
    assert cfg.view_weight == 1.0
    assert cfg.buy_weight == 3.0
