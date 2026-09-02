"""VggtPrior - the depth prior an end2end arm swaps in.

A drop-in for MotionFilter.prior_extractor; installing and restoring it is SlamRunner's job, so
nothing here can leak a patch into a later arm. Normals stay Omnidata: depth is the only variable.
"""
import torch
import torch.nn.functional as F


class VggtPrior:
    """VGGT depth + Omnidata normals. `adapter=None` is the un-adapted 'vggt_base' arm."""

    def __init__(self, cfg, adapter=None, stream_hw=None):
        from ..adapt import LoRAVGGT, aspect_lines

        # from_adapter rebuilds the structure the adapter was trained in; only the un-adapted arm
        # has nothing to read back and takes cfg.lora as written
        self.model = (LoRAVGGT.from_adapter(adapter, cfg.lora) if adapter
                      else LoRAVGGT(cfg.lora)).eval_mode()
        self.cfg = cfg
        self.hw = self.model.cfg.vggt_hw         # from_adapter may have overridden cfg.lora's
        self.label = f'{"VGGT+LoRA" if adapter else "base VGGT"} depth / Omnidata normals'

        which = f'LoRA-adapted VGGT ({adapter})' if adapter else 'base VGGT-1B (no adapter)'
        print(f'depth prior: {which} at {self.hw[1]}x{self.hw[0]}')
        print('normals    : Omnidata (unchanged, so depth is the only variable)')

        # covers the two cases the adapt stage's report cannot: an adapter trained on another
        # stream, and 'vggt_base', which has no adapter to read a size from
        if stream_hw is not None:
            for line in aspect_lines(stream_hw, self.hw, 'VggtPrior'):
                print(f'  {line}')

    def extractor(self):
        """A plain FUNCTION to install as MotionFilter.prior_extractor - never a bound method.

        Functions are descriptors, so `mf` binds as arg 0 while this VggtPrior arrives through the
        closure. A bound method or partial is not, and mf.MEAN / mf.STDV would be lost (9.3).
        """
        prior = self
        cfg = self.cfg

        @torch.amp.autocast('cuda', enabled=True)   # matches upstream prior_extractor's decorator
        @torch.no_grad()
        def prior_extractor(mf, im_tensor):
            from midas.omnidata import OmnidataModel
            from torchvision import transforms
            input_size = im_tensor.shape[-2:]

            # normals: upstream's own code. Cached on the MotionFilter, NOT on the prior - here it
            # would hold ~1 GB alive across arms and change the VRAM profile.
            if getattr(mf, 'omni_normal', None) is None:
                mf.omni_normal = OmnidataModel('normal', cfg.omni_normal_ckpt, device='cuda:0')
            resized = transforms.Resize(cfg.omni_normal_hw, antialias=True)(im_tensor).cuda()
            normal = mf.omni_normal(resized) * 2.0 - 1.0
            normal = F.interpolate(normal, input_size, mode='bicubic').float().squeeze()

            # depth: motion_filter hands us an ImageNet-NORMALISED tensor; VGGT wants [0,1] and
            # normalises internally, so undo it or it sees doubly normalised input (9.3)
            rgb = (im_tensor * mf.STDV + mf.MEAN).clamp(0, 1)
            rgb = F.interpolate(rgb, prior.hw, mode='bilinear', align_corners=False)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                depth = prior.model.predict_depth(rgb.cuda())
            # bilinear, not bicubic: bicubic can overshoot to negative depth at edges
            depth = F.interpolate(depth.float()[None, None], input_size, mode='bilinear',
                                  align_corners=False).squeeze().clamp(min=1e-3)
            return depth, normal

        return prior_extractor

    def release(self):
        """~2.5 GB, not needed by the evaluation that follows the arm's SLAM run."""
        self.model.release()


def ceil_clamp(depth, ratio):
    """depth <- min(depth, ratio * frame median) - the far-field ceiling (14).

    ratio <= 1.0 returns the input UNTOUCHED before any tensor op, so an arm with the ceiling off
    is bit-identical to one that never heard of it - which is what keeps every pre-ceiling run
    comparable. The median is taken in float32 (the ambient autocast may be fp16), the clamp is
    relative to the frame's own median so it is unit-free (Omnidata is relative depth x50, VGGT is
    in its own units), and the near half of the frame is untouched by construction.
    """
    if ratio <= 1.0:
        return depth
    d = depth.float()
    return torch.minimum(d, ratio * d.median()).to(depth.dtype)


def pedestal_shift(depth, ratio):
    """depth <- 1 / (1/depth + median(1/depth)/ratio) - the far-field PEDESTAL (14.9).

    A constant added to DISPARITY, which is the domain JDSA actually consumes: its residual is
    `rd = disps_tracker - s(p) * disps_prior` (ba.py:212), so served depth saturates at 1/b when
    the prior carries an additive disparity floor b, however far the true point is. That is not a
    hypothetical - Omnidata is a MiDaS/DPT-family AFFINE-invariant net and already carries one
    (measured: b = 0.337 x median disparity, a built-in ceiling at 2.97x the frame median), and
    JDSA's alignment is scale-only, so it never removes it. This function is that same bias made
    explicit and tunable for priors which lack it (base VGGT reads 6.97x, an adapted one 12.59x).

    UNITS - read this before comparing a @ped tag with a @ceil tag. `ratio` is the depth the
    prior saturates at in units of the frame's PRE-shift median depth, so b = median(q)/ratio.
    The shift moves the median too (every depth shrinks), so relative to the frame's own POST-
    shift median the asymptote sits at `ratio + 1`. A ceiling does not move the median - it only
    touches the top - so `@ceil1p5` and `@ped1p5` are NOT the same bound and their tags cannot be
    read against each other. `.report()` prints the realised post-shift tail for exactly this
    reason. Empirically -0.337 slope (the response of the best arm on record, omni@ceil1.5) is
    reproduced on base VGGT near ratio 1.3, not 1.5.

    Unlike ceil_clamp this map is STRICTLY MONOTONE, so it bounds the far field without flattening
    it: no pixel is collapsed onto a shared constant and the prior keeps its shape. `ratio is None`
    returns the input UNTOUCHED before any tensor op, so an arm with the pedestal off is
    bit-identical to one that never heard of it - the same property that keeps every pre-14 run
    comparable. Median in float32 (the ambient autocast may be fp16) and relative to the frame's
    own disparity, so it is unit-free (Omnidata is relative depth x50, VGGT is in its own units).
    """
    if ratio is None:
        return depth
    q = 1.0 / depth.float().clamp(min=1e-6)
    return (1.0 / (q + q.median() / ratio)).to(depth.dtype)


def soft_saturate(depth, ratio):
    """depth <- 1 / hypot(1/depth, median(1/depth)/ratio) - the SOFT far-field bound.

    The quadrature member of the one family the three tail knobs belong to: served disparity
    q' = (q^k + b^k)^(1/k) with b = median(q)/ratio, at k = 2.  ceil_clamp is k -> inf (max(q, b))
    and pedestal_shift is k = 1 (q + b); all three bound the served depth at `ratio` x the frame's
    own median depth and differ only in what they charge the NEAR field for it.  That charge is
    the reason this member exists: q >> b gives q' = q (1 + b^2/2q^2), a SECOND-order correction,
    where the pedestal's is first order.  At ratio 1.45 on kitti_00 a pixel at half the frame's
    median depth is moved 0% by the clamp, 6% here and 26% by the pedestal - and the pedestal's
    near-field cost is what its ATE pays as its ratio drops (@ped1 11.0 m against @ped2 4.1 m on
    the same prior, while their far-field assertions differ by 5%).

    Unlike ceil_clamp this map is STRICTLY MONOTONE, so no two pixels are collapsed onto a shared
    constant; unlike pedestal_shift it leaves the near field alone.  `ratio is None` returns the
    input UNTOUCHED before any tensor op, so an arm with it off is bit-identical to one that never
    heard of it.  Median in float32 (the ambient autocast may be fp16) and taken on the frame's own
    disparity, so it is unit-free.
    """
    if ratio is None:
        return depth
    q = 1.0 / depth.float().clamp(min=1e-6)
    b = q.median() / ratio
    return (1.0 / torch.sqrt(q * q + b * b)).to(depth.dtype)


def mask_far(depth, ratio):
    """depth <- 0 beyond ratio x the frame median - the far field SILENCED rather than bounded.

    The one option that is not a member of the family above: a zero depth becomes disps_prior = 0
    (depth_video.py:82), JDSA's m = (disps_prior > 0) gates the pixel out of both the residual and
    the scale grid, and ba.py:238 damps it with eta instead.  So the prior asserts nothing there,
    where a ceiling asserts "exactly ratio x median".  Those are the two readings of why bounding
    the tail helps at all, and they are what this modifier exists to separate.

    THE RATIO MUST EXCEED 1.0 (split_mods refuses otherwise): the threshold is the frame's own
    median, so ratio 1.0 deletes half the frame, and track_frontend.py:44 initialises the scale
    grid as disps.median() / disps_prior.median() over ALL pixels - zeros included - which divides
    by zero once they are the majority.  `ratio is None` returns the input UNTOUCHED.
    """
    if ratio is None:
        return depth
    d = depth.float()
    return torch.where(d > ratio * d.median(), torch.zeros_like(d), d).to(depth.dtype)


class _ServedTransform:
    """Shared plumbing for a wrapper that transforms the depth another prior SERVES (14).

    `inner` is a PLAIN function (mf, im_tensor) -> (depth, normal) - the stock extractor captured
    before SlamRunner.run installs ours (slam/stock_prior.py's recursion warning), or a
    VggtPrior's. Normals pass through untouched, so depth stays the only variable. Subclasses keep
    VggtPrior's contract - `.label`, `.extractor()`, `.release()` - and add `.stats()`/`.report()`,
    which the stage writes to `.STATS_FILE` so a run that changed nothing is detectable afterwards.
    """

    STATS_FILE = None
    # The floor a ratio must clear, MIRRORING split_mods' per-kind floor - a ceiling at or below
    # 1.0 clamps at the median and is degenerate, while a pedestal's ratio is in PRE-shift median
    # units and realises its bound at `ratio + 1` POST-shift medians, so sub-1 is meaningful there
    # and only <= 0 (a negative disparity offset) is not. Subclasses override; keep this in step
    # with split_mods or a spec the parser accepts will crash here instead, which is exactly what
    # happened when the parser floors were relaxed and this assert was not.
    RATIO_FLOOR = 1.0

    def __init__(self, inner, ratio, label, release=None, wraps=None):
        assert ratio > self.RATIO_FLOOR, (ratio, self.RATIO_FLOOR)   # split_mods refuses first
        self.inner = inner
        self.ratio = ratio
        self.label = label
        self._release = release
        self._wraps = wraps            # the transform underneath, when a spec stacked two

    def chain(self):
        """This transform and every one it wraps, outermost first.

        `inner` is a closure, not an object, so a stacked spec would otherwise lose the inner
        transform's audit trail entirely - the stage writes one stats file per link.
        """
        p = self
        while p is not None:
            yield p
            p = p._wraps

    def release(self):
        if self._release is not None:
            self._release()


class CeilingPrior(_ServedTransform):
    """Another arm's extractor with the far-field ceiling applied to the depth it serves (14).

    Built by make_prior for any '@ceil<tag>' spec. See _ServedTransform for the shared contract.
    """

    STATS_FILE = 'ceil_stats.json'

    def __init__(self, inner, ratio, label, release=None, wraps=None):
        super().__init__(inner, ratio, label, release, wraps)
        self.clip_frac = []            # per keyframe: fraction of pixels the ceiling clipped
        self.tail = []                 # per keyframe: pre-clamp (p95, p99, max) / median

    def extractor(self):
        """A plain FUNCTION for MotionFilter.prior_extractor - never a bound method (9.3)."""
        prior, ratio = self, self.ratio

        @torch.amp.autocast('cuda', enabled=True)   # matches upstream prior_extractor's decorator
        @torch.no_grad()
        def prior_extractor(mf, im_tensor):
            depth, normal = prior.inner(mf, im_tensor)
            d = depth.float()
            med = d.median().clamp(min=1e-9)
            # recorded BEFORE the clamp: this is the evidence the arm was not vacuous
            prior.clip_frac.append(float((d > ratio * med).float().mean()))
            q = torch.quantile(d.flatten(), torch.tensor([0.95, 0.99], device=d.device))
            prior.tail.append((float(q[0] / med), float(q[1] / med), float(d.max() / med)))
            return ceil_clamp(depth, ratio), normal

        return prior_extractor

    def stats(self):
        import numpy as np
        cf = np.array(self.clip_frac)
        tail = np.array(self.tail) if self.tail else np.zeros((1, 3))
        return {'ceil_ratio': self.ratio, 'n_calls': len(cf),
                'clip_frac_mean': float(cf.mean()), 'clip_frac_min': float(cf.min()),
                'clip_frac_max': float(cf.max()),
                'pre_clamp_p95_over_med': float(np.median(tail[:, 0])),
                'pre_clamp_p99_over_med': float(np.median(tail[:, 1])),
                'pre_clamp_max_over_med': float(np.median(tail[:, 2]))}

    def report(self):
        """Print the stats, loudly when the run cannot decide anything."""
        s = self.stats()
        print(f'ceiling {self.ratio:g}x median: clipped {100 * s["clip_frac_mean"]:.1f}% of '
              f'pixels on average ({100 * s["clip_frac_min"]:.1f}-'
              f'{100 * s["clip_frac_max"]:.1f}%), pre-clamp p95/p99/max over median '
              f'{s["pre_clamp_p95_over_med"]:.2f}/{s["pre_clamp_p99_over_med"]:.2f}/'
              f'{s["pre_clamp_max_over_med"]:.2f}')
        if s['clip_frac_mean'] < 0.01:
            print(f'  WARNING: the ceiling clipped under 1% of pixels - this prior asserted '
                  f'almost nothing beyond {self.ratio:g}x its median, so this arm decides '
                  f'nothing. Lower the ratio (a new ratio names a new arm) before reading the '
                  f'ATE as an answer.')
        return s


class PedestalPrior(_ServedTransform):
    """Another arm's extractor with the far-field pedestal added to the depth it serves (14.9).

    Built by make_prior for any '@ped<tag>' spec. See _ServedTransform for the shared contract and
    pedestal_shift for what `ratio` means - it is NOT the same unit as a ceiling's.
    """

    STATS_FILE = 'ped_stats.json'
    RATIO_FLOOR = 0.0              # not 1.0 - see _ServedTransform.RATIO_FLOOR and pedestal_shift

    def __init__(self, inner, ratio, label, release=None, wraps=None):
        super().__init__(inner, ratio, label, release, wraps)
        self.pre = []                  # per keyframe: pre-shift (p95, p99, max) / median
        self.post = []                 # per keyframe: the same AFTER, on the new median

    def extractor(self):
        """A plain FUNCTION for MotionFilter.prior_extractor - never a bound method (9.3)."""
        prior, ratio = self, self.ratio

        @torch.amp.autocast('cuda', enabled=True)   # matches upstream prior_extractor's decorator
        @torch.no_grad()
        def prior_extractor(mf, im_tensor):
            depth, normal = prior.inner(mf, im_tensor)
            shifted = pedestal_shift(depth, ratio)
            # BOTH tails, each against its OWN frame median: the shift moves the median, so a
            # post-shift number read against the pre-shift median would overstate the compression
            for d, into in ((depth.float(), prior.pre), (shifted.float(), prior.post)):
                med = d.median().clamp(min=1e-9)
                q = torch.quantile(d.flatten(), torch.tensor([0.95, 0.99], device=d.device))
                into.append((float(q[0] / med), float(q[1] / med), float(d.max() / med)))
            return shifted, normal

        return prior_extractor

    def stats(self):
        import numpy as np
        pre = np.array(self.pre) if self.pre else np.zeros((1, 3))
        post = np.array(self.post) if self.post else np.zeros((1, 3))
        p95_pre, p95_post = float(np.median(pre[:, 0])), float(np.median(post[:, 0]))
        return {'ped_ratio': self.ratio, 'n_calls': len(self.pre),
                'pre_p95_over_med': p95_pre,
                'pre_p99_over_med': float(np.median(pre[:, 1])),
                'pre_max_over_med': float(np.median(pre[:, 2])),
                'post_p95_over_med': p95_post,
                'post_p99_over_med': float(np.median(post[:, 1])),
                'post_max_over_med': float(np.median(post[:, 2])),
                'p95_shrink': float(1.0 - p95_post / max(p95_pre, 1e-9))}

    def report(self):
        """Print the stats, loudly when the run cannot decide anything."""
        s = self.stats()
        print(f'pedestal {self.ratio:g}x median (pre-shift): served tail p95/p99/max over the '
              f"frame's own median {s['pre_p95_over_med']:.2f}/{s['pre_p99_over_med']:.2f}/"
              f"{s['pre_max_over_med']:.2f} -> {s['post_p95_over_med']:.2f}/"
              f"{s['post_p99_over_med']:.2f}/{s['post_max_over_med']:.2f}, "
              f"p95 shrank {100 * s['p95_shrink']:.1f}%")
        if s['p95_shrink'] < 0.01:
            print(f'  WARNING: the pedestal shrank the served p95 by under 1% - at ratio '
                  f'{self.ratio:g} this prior was already flatter than the bound, so this arm '
                  f'decides nothing. Lower the ratio (a new ratio names a new arm) before '
                  f'reading the ATE as an answer.')
        return s


class SoftPrior(_ServedTransform):
    """Another arm's extractor with the SOFT far-field bound on the depth it serves.

    Built by make_prior for any '@soft<tag>' spec. See soft_saturate for the family this is the
    k = 2 member of, and _ServedTransform for the shared contract.
    """

    STATS_FILE = 'soft_stats.json'
    RATIO_FLOOR = 0.0              # bounds without clipping, so sub-1 is meaningful (split_mods)

    def __init__(self, inner, ratio, label, release=None, wraps=None):
        super().__init__(inner, ratio, label, release, wraps)
        self.pre = []                  # per keyframe: pre-shift (p95, p99, max) / median
        self.post = []                 # per keyframe: the same AFTER, on the new median
        self.near_cost = []            # per keyframe: how much the NEAR half was moved

    def extractor(self):
        """A plain FUNCTION for MotionFilter.prior_extractor - never a bound method (9.3)."""
        prior, ratio = self, self.ratio

        @torch.amp.autocast('cuda', enabled=True)   # matches upstream prior_extractor's decorator
        @torch.no_grad()
        def prior_extractor(mf, im_tensor):
            depth, normal = prior.inner(mf, im_tensor)
            softened = soft_saturate(depth, ratio)
            d0, d1 = depth.float(), softened.float()
            for d, into in ((d0, prior.pre), (d1, prior.post)):
                med = d.median().clamp(min=1e-9)
                q = torch.quantile(d.flatten(), torch.tensor([0.95, 0.99], device=d.device))
                into.append((float(q[0] / med), float(q[1] / med), float(d.max() / med)))
            # the number this family is chosen on: what the near field paid for the far bound
            near = d0 < d0.median()
            prior.near_cost.append(float((1.0 - d1[near] / d0[near].clamp(min=1e-9)).mean()))
            return softened, normal

        return prior_extractor

    def stats(self):
        import numpy as np
        pre = np.array(self.pre) if self.pre else np.zeros((1, 3))
        post = np.array(self.post) if self.post else np.zeros((1, 3))
        p95_pre, p95_post = float(np.median(pre[:, 0])), float(np.median(post[:, 0]))
        return {'soft_ratio': self.ratio, 'n_calls': len(self.pre),
                'pre_p95_over_med': p95_pre,
                'pre_p99_over_med': float(np.median(pre[:, 1])),
                'pre_max_over_med': float(np.median(pre[:, 2])),
                'post_p95_over_med': p95_post,
                'post_p99_over_med': float(np.median(post[:, 1])),
                'post_max_over_med': float(np.median(post[:, 2])),
                'p95_shrink': float(1.0 - p95_post / max(p95_pre, 1e-9)),
                'near_half_shrink': float(np.mean(self.near_cost)) if self.near_cost else 0.0}

    def report(self):
        """Print the stats, loudly when the run cannot decide anything."""
        s = self.stats()
        print(f'soft bound {self.ratio:g}x median: served tail p95/p99/max over the frame\'s own '
              f"median {s['pre_p95_over_med']:.2f}/{s['pre_p99_over_med']:.2f}/"
              f"{s['pre_max_over_med']:.2f} -> {s['post_p95_over_med']:.2f}/"
              f"{s['post_p99_over_med']:.2f}/{s['post_max_over_med']:.2f}, p95 shrank "
              f"{100 * s['p95_shrink']:.1f}%, near half paid "
              f"{100 * s['near_half_shrink']:.1f}%")
        if s['p95_shrink'] < 0.01:
            print(f'  WARNING: the soft bound shrank the served p95 by under 1% - at ratio '
                  f'{self.ratio:g} this prior was already flatter than the bound, so this arm '
                  f'decides nothing. Lower the ratio (a new ratio names a new arm) before '
                  f'reading the ATE as an answer.')
        return s


class MaskPrior(_ServedTransform):
    """Another arm's extractor with the far field DELETED from the depth it serves.

    Built by make_prior for any '@mask<tag>' spec. See mask_far for what a zero depth means to
    JDSA, and _ServedTransform for the shared contract.
    """

    STATS_FILE = 'mask_stats.json'

    def __init__(self, inner, ratio, label, release=None, wraps=None):
        super().__init__(inner, ratio, label, release, wraps)
        self.masked = []               # per keyframe: fraction of pixels deleted

    def extractor(self):
        """A plain FUNCTION for MotionFilter.prior_extractor - never a bound method (9.3)."""
        prior, ratio = self, self.ratio

        @torch.amp.autocast('cuda', enabled=True)   # matches upstream prior_extractor's decorator
        @torch.no_grad()
        def prior_extractor(mf, im_tensor):
            depth, normal = prior.inner(mf, im_tensor)
            d = depth.float()
            prior.masked.append(float((d > ratio * d.median()).float().mean()))
            return mask_far(depth, ratio), normal

        return prior_extractor

    def stats(self):
        import numpy as np
        m = np.array(self.masked) if self.masked else np.zeros(1)
        return {'mask_ratio': self.ratio, 'n_calls': len(self.masked),
                'masked_frac_mean': float(m.mean()), 'masked_frac_min': float(m.min()),
                'masked_frac_max': float(m.max())}

    def report(self):
        """Print the stats, loudly when the run cannot decide anything - or is unsafe."""
        s = self.stats()
        print(f'mask {self.ratio:g}x median: deleted {100 * s["masked_frac_mean"]:.1f}% of pixels '
              f'on average ({100 * s["masked_frac_min"]:.1f}-{100 * s["masked_frac_max"]:.1f}%)')
        if s['masked_frac_mean'] < 0.01:
            print(f'  WARNING: the mask deleted under 1% of pixels - this prior asserted almost '
                  f'nothing beyond {self.ratio:g}x its median, so this arm decides nothing.')
        if s['masked_frac_max'] > 0.45:
            print(f'  WARNING: one keyframe lost {100 * s["masked_frac_max"]:.1f}% of its pixels. '
                  f'track_frontend.py:44 seeds the scale grid with disps_prior.median() over ALL '
                  f'pixels, so past 50% that seed is 0 and the grid starts at infinity.')
        return s


# The spec modifier vocabulary, resolved to what serves it. config.py owns the SYNTAX (which tags
# parse, and MOD_ORDER); this owns the BEHAVIOUR. Keep the two key sets equal - wrap_mods asserts.
_MOD_PRIORS = {'ceil': CeilingPrior, 'soft': SoftPrior, 'ped': PedestalPrior, 'mask': MaskPrior}


def wrap_mods(inner, mods, label, release=None):
    """Serve `inner` through every modifier in `mods`, applied in MOD_ORDER. None if `mods` is {}.

    Returns the OUTERMOST wrapper; `.chain()` walks back down it. The caller decides what an
    unmodified prior is - stock Omnidata is None (the untouched upstream path), a VggtPrior is
    itself - so this deliberately returns None rather than guessing. `release` is attached to the
    outermost link, because that is the object the stage calls release() on.
    """
    from .config import MOD_ORDER
    assert set(_MOD_PRIORS) == set(MOD_ORDER), (sorted(_MOD_PRIORS), sorted(MOD_ORDER))
    chain = [k for k in MOD_ORDER if k in mods]
    prior = None
    for i, kind in enumerate(chain):
        prior = _MOD_PRIORS[kind](inner if prior is None else prior.extractor(), mods[kind], label,
                                  release=release if i == len(chain) - 1 else None, wraps=prior)
    return prior


def mods_label(mods, sep=' + '):
    """The phrase a modified arm's label carries, e.g. ' + ceil 1.5x median + pedestal 2x median'.

    One spelling, so the end2end and priortest stages cannot drift apart in what they call an arm.
    """
    from .config import MOD_ORDER
    words = {'ceil': 'ceil', 'soft': 'soft bound', 'ped': 'pedestal',
             'mask': 'mask beyond'}
    return ''.join(f'{sep}{words[k]} {mods[k]:g}x median' for k in MOD_ORDER if k in mods)
