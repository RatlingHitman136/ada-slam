"""Where this repository is, and the sys.path bootstrap every package here needs.

`ada-slam/` has a hyphen in its name, so it is never imported as a package - it is a directory
placed on sys.path, exactly like `hislam2/`, after which `from adapt import ...`, `from slam
import ...` and `from common import ...` work. Each package's __init__ calls bootstrap() so that
`import slam` works from any cwd and, more importantly, inside a spawned child that re-imports the
reader's module.

Deliberately stdlib-only and side-effect-free apart from the sys.path inserts: a spawned child
imports this before anything else, and it must cost nothing.
"""
import os
import sys

ADA_SLAM = os.path.dirname(os.path.abspath(__file__))       # <repo>/ada-slam
ROOT = os.path.dirname(ADA_SLAM)                            # <repo>
HISLAM2 = os.path.join(ROOT, 'hislam2')
VGGT = os.path.join(ROOT, 'thirdparty/vggt')


def bootstrap(*extra):
    """Put ADA_SLAM (and any extra dirs) on sys.path, front, without duplicating."""
    for p in (ADA_SLAM, *extra):
        if p not in sys.path:
            sys.path.insert(0, p)


def ensure(*dirs):
    """makedirs(exist_ok=True) for each, returning them so callers can inline the call."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs[0] if len(dirs) == 1 else dirs
