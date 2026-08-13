"""Continuous adaptation: ONE SLAM run in which the depth prior learns as the map is built (13).

The offline track is three stages - extract dumps SLAM depth, adapt trains on the dump, end2end
runs a second SLAM pass with the frozen adapter. This is all of it at once: every keyframe the
tracker settles becomes a training sample for the very prior serving the next one.

    run_online_adapt(runner, ONLINE, END2END, adapt_out, ckpt_dir, arm_out, CONFIG,
                     length, buffer, split_at, stream_hw=..., init_adapter=...)

hislam2/ is untouched. The hook is that SlamRunner installs a prior as a plain function on
MotionFilter, so `mf.video` reaches the tracker's whole shared state from inside the extractor -
see target.py for what is read off it and prior.py for when.
"""
from .config import OnlineConfig
from .stage import run_online_adapt

__all__ = ['OnlineConfig', 'run_online_adapt']
