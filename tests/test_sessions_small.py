import pandas as pd
from src.sessions import build_sessions


def test_build_sessions_30min_gap():
    # timestamps in milliseconds
    df = pd.DataFrame(
        {
            "visitorid": [1, 1, 1],
            "timestamp": [
                0,                  # 0 min
                10 * 60 * 1000,      # 10 min -> same session
                70 * 60 * 1000,      # 70 min -> gap 60 min -> new session
            ],
            "event": ["view", "view", "view"],
            "itemid": [101, 102, 103],
        }
    )

    out = build_sessions(df, gap_minutes=30)

    # first two should be same session
    assert out.loc[0, "session_id"] == out.loc[1, "session_id"]

    # third should be a new session
    assert out.loc[2, "session_id"] != out.loc[1, "session_id"]

