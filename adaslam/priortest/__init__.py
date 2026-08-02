"""Test kind 2: the same generators scored against GT depth, no SLAM run (9.2.2)."""
from .config import PriorTestConfig
from .stage import run_prior_test

__all__ = ['PriorTestConfig', 'run_prior_test']
