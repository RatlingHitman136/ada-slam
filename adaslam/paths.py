"""Where this repository is, and the sys.path entries the non-installable roots need.

Stdlib-only and side-effect-free apart from the inserts: every adaslam.* import goes through here.
"""
import os
import sys

ADA_SLAM = os.path.dirname(os.path.abspath(__file__))       # <repo>/adaslam
ROOT = os.path.dirname(ADA_SLAM)                            # <repo>
HISLAM2 = os.path.join(ROOT, 'hislam2')
VGGT = os.path.join(ROOT, 'thirdparty/vggt')


def bootstrap(*dirs):
    """Put each of `dirs` on sys.path, front, without duplicating.

    Exactly what it is given: ADA_SLAM must NOT go on sys.path, or `adapt` and `adaslam.adapt`
    become two module objects with two LoRAConfig classes that fail isinstance (9.5).
    """
    for p in dirs:
        if p not in sys.path:
            sys.path.insert(0, p)
