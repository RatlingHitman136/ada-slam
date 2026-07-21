"""Baseline: HI-SLAM2 on a full sequence with the stock Omnidata depth prior, then report.

This is the control arm of the A/B. It patches nothing, so the run is byte-identical to demo.py;
the only addition is the evaluation and the seen/unseen split report.

  python scripts/run_full_omnidata.py --output outputs/ab/room0_omnidata

Pair it with run_full_vggt.py, which differs only in where the depth prior comes from.
"""
import os    # nopep8
import sys   # nopep8
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # nopep8

from _full_run_common import main

if __name__ == '__main__':
    main('Omnidata depth (baseline)')
