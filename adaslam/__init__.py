"""The VGGT depth-prior track's own code (ARCHITECTURE.md §9), as one package.

    from adaslam.adapt import LoRAVGGT
    from adaslam.extract.export import load_export

Importing any submodule runs this file first - Python imports parent packages before children -
which is what puts hislam2/ and thirdparty/vggt on sys.path. Those two are the roots that cannot
be installed (§9.5: hislam2/ has no top-level __init__.py and imports flatly among itself, and
vggt is vendored because its requirements would downgrade torch), and this is the only place in
the repo besides demo.py that adds anything to sys.path.

Not needed in a torch.multiprocessing child: spawn copies the parent's sys.path into the child
(multiprocessing/spawn.py:173, 228-229) before it re-imports __main__.
"""
from .paths import HISLAM2, VGGT, bootstrap

bootstrap(HISLAM2, VGGT)
