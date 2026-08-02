"""PriorProbe - running a depth prior exactly as MotionFilter would, with no SLAM run.

It must call the SAME function a real arm calls, or it reports a number no arm produces. The stock
prior IS a MotionFilter method and only this package may import it, hence the probe living here:

    MotionFilter.prior_extractor(host, tensor)      # stock: a plain function, host is arg 0
    VggtPrior.extractor()(host, tensor)             # an arm: same shape, by construction

`host` stands in for the MotionFilter instance - both extractors want only MEAN, STDV and model
cache slots off it, and keeping one per arm stops the models reloading every frame.
"""
import types

import torch

from .stream import load_calib, load_frame

# Must match motion_filter.py:44-45. Copied, not imported: MotionFilter sets them in __init__, and
# constructing one needs a DroidNet, a DepthVideo and a CUDA context.
_MEAN = (0.485, 0.456, 0.406)
_STDV = (0.229, 0.224, 0.225)


class PriorProbe:
    """Depth from one prior generator, frame by frame, at tracking resolution.

    `prior=None` is the stock Omnidata path. Lazy: nothing is loaded until the first depth() call,
    so building one costs nothing and a skipped arm costs nothing.
    """

    def __init__(self, cfg, prior=None):
        self.cfg = cfg
        self.prior = prior
        self._calib = None
        self._host = None
        self._extract = None

    def _ensure(self):
        if self._extract is not None:
            return
        from motion_filter import MotionFilter
        dev = 'cuda'
        self._calib = load_calib(self.cfg)
        # omni_dep / omni_normal are the slots both extractors cache their models on
        self._host = types.SimpleNamespace(
            MEAN=torch.as_tensor(_MEAN, device=dev)[:, None, None],
            STDV=torch.as_tensor(_STDV, device=dev)[:, None, None],
            omni_dep=None, omni_normal=None)
        self._extract = (MotionFilter.prior_extractor if self.prior is None
                         else self.prior.extractor())

    @torch.no_grad()
    def depth(self, path):
        """Metric depth (H, W) float32 at tracking resolution for one colour file.

        The normals are discarded: a depth-only path would be a second copy of the depth code.
        """
        self._ensure()
        image, _ = load_frame(self.cfg, path, *self._calib)
        # exactly motion_filter.py:88-89, which is what prior_extractor is written against
        t = torch.as_tensor(image).permute(2, 0, 1)[None].cuda().float().div_(255.0)
        t = t.sub_(self._host.MEAN).div_(self._host.STDV)
        depth, _normal = self._extract(self._host, t)
        return depth.float().cpu().numpy()

    def release(self):
        """The Omnidata pair is ~1.4 GB and the VGGT model ~2.5 GB; neither outlives its arm."""
        if self.prior is not None:
            self.prior.release()
        self._host = self._extract = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
