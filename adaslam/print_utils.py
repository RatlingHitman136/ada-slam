"""Formatting output for a human. Nothing here computes anything.

Stdlib only - no torch, no cv2 - so importing it costs nothing and a report can be re-run on a
machine with neither. That matters: both report modules are pure formatting over dicts on disk, and
re-reading a finished comparison should never need a GPU box.

Kept out of runtime.py, which is process and shared-workstation hygiene (VRAM, subprocesses, fd
limits) and whose docstring is a measurement of CUDA IPC leakage. free_vram() and gpu_gate() print
too, but they belong there: they do work and report on it, rather than formatting someone else's
numbers.
"""
import contextlib
import sys


def banner(title):
    """A stage or arm boundary in the log."""
    print(f'\n{"=" * 78}\n=== {title}\n{"=" * 78}')


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


@contextlib.contextmanager
def tee(path):
    """Print to stdout and to a file at once - export.txt is read back by other tooling."""
    with open(path, 'w') as f:
        with contextlib.redirect_stdout(_Tee(sys.stdout, f)):
            yield


def delta_header(labels, width=12, name_width=26):
    """The column headers and rule above a block of delta_row()s.

    Shared so the two comparison tables print identically by construction rather than by two people
    happening to write the same f-strings. `width` is normally computed from the arm names, which
    are as long as an adapter's experiment name plus a checkpoint suffix.
    """
    print(f"  {'metric':<{name_width}}{labels[0]:>{width}}"
          + ''.join(f'{lbl:>{width}}{"delta":>10}' for lbl in labels[1:]))
    print('  ' + '-' * (name_width + width + (width + 10) * (len(labels) - 1)))


def delta_row(name, values, lower_better=True, width=12, name_width=26):
    """One comparison row: the baseline absolute, every later column absolute + signed delta.

    `values[0]` is the baseline. Each later value gets a `+` when it beats the baseline and a `-`
    when it loses, judged by `lower_better`; an exact tie gets a blank, not a `+`. `None` prints as
    `n/a`, and a `None` BASELINE prints nothing at all - there is nothing to compare against.

    Shared by both report modules. Their table layouts stay separate, because those genuinely
    differ - the prior test repeats its table per population and can mark an arm whose split is not
    its own, which the end2end comparison has no notion of - but the row itself was the same code
    twice.
    """
    if values[0] is None:
        return
    line = f'  {name:<{name_width}}{values[0]:>{width}.4f}'
    for v in values[1:]:
        if v is None:
            line += f'{"n/a":>{width}}{"":>11}'
            continue
        d = v - values[0]
        mark = ' ' if abs(d) < 1e-9 else ('+' if (d < 0) == lower_better else '-')
        line += f'{v:>{width}.4f}{d:>+9.4f} {mark}'
    print(line)
