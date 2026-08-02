"""Writing a derived tracking YAML - HI-SLAM2 schema knowledge, hence slam/.

Only overridden keys need to appear: load_config resolves `inherit_from` recursively. The VALUES
are the caller's policy and live in its config.
"""
import os


def write_tracking_config(out, base_config, motion_thresh=None, init_thresh=None,
                          keyframe_thresh=None, covis_thresh=None, name='extract_config.yaml'):
    """Write `out/name` inheriting from base_config, overriding whichever knobs are not None."""
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
