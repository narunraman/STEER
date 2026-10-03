# Entry point for Inspect: makes `inspect eval steer_bench/steer` and `steer_bench/steer_me` work.
from .tasks import steer, steer_me

__all__ = ["steer", "steer_me"]
