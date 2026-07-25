"""Writing a derived tracking YAML.

This is HI-SLAM2 config-schema knowledge - which section each keyframe knob lives under, and that
load_config() resolves `inherit_from` recursively and merges, so only the overridden keys need to
appear. It sits in slam/ for that reason; the VALUES are a caller's policy and live in its config
(ExtractConfig, for the one caller there is).
"""
import os


def write_tracking_config(out, base_config, motion_thresh=None, init_thresh=None,
                          keyframe_thresh=None, covis_thresh=None, name='extract_config.yaml'):
    """Write `out/name` inheriting from base_config, overriding whichever knobs are not None.

    Doubles as a record of what the run was actually told to do. Returns the path.
    """
    import yaml
    tracking = {}
    for section, keys in (('motion_filter', (('thresh', motion_thresh),
                                             ('init_thresh', init_thresh))),
                          ('frontend', (('keyframe_thresh', keyframe_thresh),)),
                          ('backend', (('covis_thresh', covis_thresh),))):
        vals = {k: v for k, v in keys if v is not None}
        if vals:
            tracking[section] = vals

    cfg = {'inherit_from': os.path.abspath(base_config)}  # absolute: load_config resolves vs cwd
    if tracking:
        cfg['Tracking'] = tracking
    os.makedirs(out, exist_ok=True)
    path = f'{out}/{name}'
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f'extract config: {path}  ({tracking if tracking else "no overrides"})')
    return path
