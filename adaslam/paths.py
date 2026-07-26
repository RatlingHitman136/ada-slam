"""Where this repository is, and the sys.path entries the non-installable roots need.

Two of the three source roots this repo runs on cannot be installed into the venv: `hislam2/` has
no top-level __init__.py and its own code imports flatly among itself (`from geom.ba import ...`),
and `thirdparty/vggt` is vendored rather than pip-installed because its requirements pin
torch==2.3.1 / numpy==1.26.1. `adaslam/__init__.py` calls bootstrap() with both, once, and because
Python imports a parent package before any of its children that covers every `adaslam.*` import -
including in a spawned child, which inherits the parent's sys.path anyway
(multiprocessing/spawn.py:173, 228-229).

Deliberately stdlib-only and side-effect-free apart from those inserts: every import of anything in
this package goes through here, so it must cost nothing.
"""
import os
import sys

ADA_SLAM = os.path.dirname(os.path.abspath(__file__))       # <repo>/adaslam
ROOT = os.path.dirname(ADA_SLAM)                            # <repo>
HISLAM2 = os.path.join(ROOT, 'hislam2')
VGGT = os.path.join(ROOT, 'thirdparty/vggt')


def bootstrap(*dirs):
    """Put each of `dirs` on sys.path, front, without duplicating.

    Exactly what it is given and nothing implicit - ADA_SLAM in particular must NOT go on sys.path.
    This is a package now, so putting its own directory there as well would make `adapt` and
    `adaslam.adapt` two distinct module objects with separate globals: two LoRAConfig classes that
    fail isinstance against each other.
    """
    for p in dirs:
        if p not in sys.path:
            sys.path.insert(0, p)
