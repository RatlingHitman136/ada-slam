"""-c NAME: a driver's PARAMETERS block read from run_configs/NAME.yaml instead of its literals.

A queued job cannot edit the driver between launches, and a bigger GPU is a different BUFFER, a
different MIN_FREE_VRAM_MB and usually a different window - so every knob a run varies has to be
settable from a file. This is that file's reader and nothing else: it knows no parameter's meaning,
it only replaces literals by name.

WITHOUT -c nothing changes - the driver's literals stand, exactly as they did. WITH -c the file is
the WHOLE parameter set: a key the driver never asks for and a parameter the file never states are
both errors (done()), because a queued run that silently kept one default is a wasted run.

Read at MODULE scope, the single exception to 9.5's "no disk in the PARAMETERS block". That rule
exists because spawn re-executes the driver in every child; here it is harmless and in fact
necessary - multiprocessing/spawn.py copies sys.argv into the child and chdirs it to the parent's
cwd BEFORE re-importing the module, so the child reads the same file and rebuilds the same values.
"""
import argparse
import os
from dataclasses import fields, replace

RUN_CONFIG_DIR = 'run_configs'


def run_config(root, prefix='', doc=None):
    """Parse -c and load the file, or an empty RunConfig when it was not given.

    `root` is the repo root: this runs BEFORE main()'s chdir, so a NAME resolves against the
    driver's own run_configs/ rather than the directory the job was launched from. `prefix` is the
    driver's filename convention ('live_'), refused rather than assumed so a queue cannot hand one
    driver's config to another.
    """
    ap = argparse.ArgumentParser(description=(doc or '').split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-c', '--config', metavar='NAME',
                    help=f'a run config in {RUN_CONFIG_DIR}/ - NAME, NAME.yaml or a path. It '
                         f"states EVERY parameter; without it this file's literals are used")
    name = ap.parse_args().config
    if name is None:
        return RunConfig({})

    path = name if os.path.exists(name) else os.path.join(
        root, RUN_CONFIG_DIR, name if name.endswith('.yaml') else f'{name}.yaml')
    if not os.path.exists(path):
        raise SystemExit(f'run config not found: {path}')
    if prefix and not os.path.basename(path).startswith(prefix):
        raise SystemExit(f'{path}: this driver reads {prefix}*.yaml only - a config named for '
                         f'another driver would set knobs this one does not have')
    import yaml
    with open(path) as f:
        values = yaml.safe_load(f) or {}
    if not isinstance(values, dict):
        raise SystemExit(f'{path}: a run config is a mapping of PARAMETER: value')
    return RunConfig(values, path)


class RunConfig:
    """One parsed run config. Empty (`path` None) is the no -c case, where every literal stands."""

    def __init__(self, values, path=None):
        self._v = _tuples(values)
        self._want = set()      # every parameter the driver asked for, 'KEY' or 'section.field'
        self.path = path

    def __call__(self, key, default):
        """`X = P('X', <the literal>)` - the file's X, or the literal when there is no file."""
        self._want.add(key)
        if self.path is None or key not in self._v:
            return default      # a missing key is reported by done(), together with the others
        return _checked(key, self._v[key], default)

    def over(self, section, cfg, fixed=()):
        """A dataclass literal with the file's `section:` mapping applied over its fields.

        `fixed` names the fields the block above feeds from a top-level key (ONLINE's stream_res is
        STREAM_RES); refused here, so a knob has exactly one spelling in the file.
        """
        names = [f.name for f in fields(cfg) if f.name not in fixed]
        self._want.update(f'{section}.{n}' for n in names)
        if self.path is None:
            return cfg
        d = self._v.get(section) or {}
        if not isinstance(d, dict):
            raise SystemExit(f'{self.path}: {section} must be a mapping of field: value')
        over = {}
        for k, v in d.items():
            if k in fixed:
                raise SystemExit(f'{self.path}: {section}.{k} is fed by a top-level key of this '
                                 f'driver; state it there instead')
            if k not in names:
                raise SystemExit(f'{self.path}: {section}.{k} is not a field of '
                                 f'{type(cfg).__name__} - {sorted(names)}')
            over[k] = _checked(f'{section}.{k}', v, getattr(cfg, k))
        return replace(cfg, **over)

    def done(self):
        """The file states exactly this driver's parameters. Checked once, after the last one.

        Both directions, and neither is pedantry: a key nothing read is a typo that silently ran a
        default, and an omitted parameter is a knob the queue did not set and did not know it was
        not setting.
        """
        if self.path is None:
            return
        have = set(_flat(self._v))
        missing, unknown = sorted(self._want - have), sorted(have - self._want)
        if missing or unknown:
            raise SystemExit(
                f"{self.path} does not state this driver's parameters:"
                + (f'\n  MISSING ({len(missing)}): {missing}' if missing else '')
                + (f'\n  UNKNOWN ({len(unknown)}): {unknown}' if unknown else '')
                + f'\n  a run config states every parameter - copy {RUN_CONFIG_DIR}/'
                  f'live_default.yaml and edit it')


def _tuples(v):
    """YAML lists -> tuples, so a file value is exactly what the literal would have been."""
    if isinstance(v, list):
        return tuple(_tuples(x) for x in v)
    if isinstance(v, dict):
        return {k: _tuples(x) for k, x in v.items()}
    return v


def _flat(values):
    """{'STOP': 1000, 'online': {'lag': 5}} -> {'STOP', 'online.lag'} - done()'s comparison shape."""
    out = {}
    for k, v in values.items():
        out.update({f'{k}.{kk}': vv for kk, vv in v.items()} if isinstance(v, dict) else {k: v})
    return out


def _checked(key, value, default):
    """Refuse a value whose type cannot be what the literal is.

    YAML 1.1 reads `1e-4` as a STRING (PyYAML's float resolver wants the dot), and a string lr is a
    TypeError a thousand optimiser steps later, in a child, at 2 a.m. Unquoted yes/no/on/off are
    booleans for the same reason.
    """
    if value is None or default is None:
        return value                       # None is a stated instruction here (STOP, VGGT_HW)
    if isinstance(default, bool):
        ok = isinstance(value, bool)
    elif isinstance(default, (int, float)):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif isinstance(default, (str, tuple)):
        ok = isinstance(value, type(default))
    else:
        return value
    if not ok:
        raise SystemExit(f'{key}: expected {type(default).__name__}, got {value!r} '
                         f'({type(value).__name__}). YAML reads 1e-4 as a string - write 1.0e-4')
    return value
