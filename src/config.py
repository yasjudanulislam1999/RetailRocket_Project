from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()  # loads variables from a local .env file (if it exists)


def _get_float(name: str, default: float) -> float:
    """
    Read a float from environment variables.
    Example: VIEW_WEIGHT="1.0"
    """
    value = os.getenv(name)
    return float(value) if value is not None else default


def _get_int(name: str, default: int) -> int:
    """
    Read an int from environment variables.
    Example: TOPK="50"
    """
    value = os.getenv(name)
    return int(value) if value is not None else default


@dataclass(frozen=True)
class Config:
    """
    One place for simple project settings.
    """
    topk: int = _get_int("TOPK", 50)
    view_weight: float = _get_float("VIEW_WEIGHT", 1.0)
    buy_weight: float = _get_float("BUY_WEIGHT", 3.0)
