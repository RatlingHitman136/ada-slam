"""Formatting output for a human. Nothing here computes anything.

Stdlib only, so a finished comparison can be reprinted on a machine with no torch and no GPU.
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
    """The column headers and rule above a block of delta_row()s."""
    print(f"  {'metric':<{name_width}}{labels[0]:>{width}}"
          + ''.join(f'{lbl:>{width}}{"delta":>10}' for lbl in labels[1:]))
    print('  ' + '-' * (name_width + width + (width + 10) * (len(labels) - 1)))


def delta_row(name, values, lower_better=True, width=12, name_width=26):
    """One comparison row: values[0] absolute, every later column absolute + signed delta.

    A later value gets '+' when it beats the baseline, '-' when it loses, blank on an exact tie.
    None prints as n/a; a None baseline prints nothing - there is nothing to compare against.
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
