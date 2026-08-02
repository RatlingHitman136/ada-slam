"""The VGGT depth-prior track's own code (ARCHITECTURE.md 9), as one package.

Importing any submodule runs this first, which is what puts hislam2/ and thirdparty/vggt on
sys.path. A spawn child inherits sys.path, so nothing here has to run again there.
"""
from .paths import HISLAM2, VGGT, bootstrap

bootstrap(HISLAM2, VGGT)
