"""Stage 1: HI-SLAM2's own depth, exported as training data (9.2.1).

The experiment directory's top level is the handoff to adapt; full/ is the untouched SLAM run and
can be deleted afterwards. Loading is split from writing (export.load_export takes full/, the
others take its arrays), so an existing slam_depth.npz can be re-exported without re-running SLAM.
"""
from .config import ExtractConfig
from .stage import run_extract

__all__ = ['ExtractConfig', 'run_extract']
