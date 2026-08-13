"""The upstream Omnidata depth prior, as a plain function.

Here rather than next to its one caller because only `adaslam/slam/` may import `motion_filter`
(9.3) - the stock prior IS a MotionFilter method, which is the same reason prior_probe.py lives
here.
"""


def stock_prior_extractor():
    """`MotionFilter.prior_extractor` as it is RIGHT NOW - Omnidata depth + Omnidata normals.

    CAPTURE THIS BEFORE SlamRunner.run INSTALLS A PRIOR. run() overwrites the class attribute
    (runner.py:81-83), so a caller that fetches it lazily from inside its own installed extractor
    fetches *itself* and recurses forever. The online stage takes it in __init__ for that reason.
    """
    from motion_filter import MotionFilter
    return MotionFilter.prior_extractor
