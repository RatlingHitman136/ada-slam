"""run_online_adapt - the whole single-stage experiment: one SLAM run that adapts as it tracks.

It is stage 2 and stage 3 at once. What comes out is what those two stages produce separately:

    outputs/adapt/<SCENE>/<NAME>/                adapter.safetensors, config.json, train_log.json
                                checkpoints/epoch_NNN/
    outputs/test/end2end/<SCENE>/<NAME>_live/    traj_full.txt, results.json, evo/

so the adapter is a normal handoff directory (testable frozen, usable as ADAPT_INIT) and the arm is
a normal end2end arm (compare(), ate_over_time.py, export_end2end_results.py all read it).

The _live suffix on the arm is not decoration - see common.py:ONLINE_ARM_SUFFIX.
"""
import json
import os
import time

from ..adapt.stage import init_adapter_path
from ..common import ADAPTER_FILE
from ..end2end.metrics import evaluate
from ..end2end.report import print_report
from ..end2end.stage import cached_results
from ..print_utils import banner
from ..runtime import free_vram

from .prior import OnlineVggtPrior

TRAIN_LOG = 'train_log.json'
GATE_LOG = 'gate_log.json'      # every arrival the loss gate saw, trained or skipped


def make_record(arm_out, split_at):
    """`f(trainer, unit) -> dict` - what a save of `unit` records, checkpoints included.

    Everything about the training itself comes from trainer.stats(); this adds only what the
    trainer cannot know. `scene` is the ARM's output directory rather than an extract's: the SLAM
    run that produced the supervision is this same run, and there is no export to point at.
    """
    def record(trainer, unit):
        return {**trainer.stats(), 'saved_epoch': unit, 'split_at': split_at, 'scene': arm_out}
    return record


def run_online_adapt(runner, online_cfg, e2e_cfg, adapt_out, ckpt_dir, arm_out, arm_config,
                     length, buffer, split_at, stream_hw=None, init_adapter=None,
                     skip_existing=False):
    """One continuously-adapting SLAM run, saved as an adapter and scored as an arm.

    Every path is an argument and none is read out of a global (9.5). `init_adapter` is the adapt
    directory to CONTINUE from, or None for stock VGGT-1B - the same vocabulary an END2END_PRIORS
    entry uses.
    """
    banner(f'online adapt  -> {adapt_out}   |   arm -> {arm_out}')
    adapter_file = f'{adapt_out}/{ADAPTER_FILE}'

    # Two-level cache, as end2end/stage.py's: the RUN is the expensive half and produces both
    # artifacts at once, so it is reused only when both survived. Re-scoring at another split is
    # one evo_ape over a trajectory already on disk.
    done = os.path.exists(adapter_file) and os.path.exists(f'{arm_out}/traj_full.txt')
    if skip_existing and done:
        res = cached_results(arm_out, split_at)
        if res is not None:
            print(f'{adapter_file} and results at split_at={split_at} exist - skipping')
            return res
        print(f'{arm_out}/traj_full.txt exists - reusing the run, re-scoring at '
              f'split_at={split_at}')
        cfg_path = f'{adapt_out}/config.json'
        label = json.load(open(cfg_path)).get('label', 'VGGT adapted ONLINE') \
            if os.path.exists(cfg_path) else 'VGGT adapted ONLINE'
        res = evaluate(arm_out, label, split_at, e2e_cfg)
        print_report(res)
        return res

    init = init_adapter_path(init_adapter)
    print(f'starting from {init if init else "stock VGGT-1B (no adapter)"}')
    os.makedirs(adapt_out, exist_ok=True)

    t0 = time.time()
    # runner.cfg IS the SlamConfig, so the window's start comes from its one definition rather
    # than being mirrored into OnlineConfig and kept in sync by hand
    prior = OnlineVggtPrior(e2e_cfg, online_cfg, adapter=init, stream_hw=stream_hw,
                            ckpt_dir=ckpt_dir, record=make_record(arm_out, split_at),
                            frame_offset=runner.cfg.start)
    label = prior.label
    try:
        # SlamRunner installs and restores the prior, so nothing leaks into a later arm even if
        # this raises. gtdepthdir stays None - nothing scores renders any more (9.3, 11).
        n_kf = runner.run(arm_out, arm_config, length, buffer,
                          gtdepthdir=None, prior=prior).n_kf
        print(f'\nSLAM done in {time.time()-t0:.0f}s, {n_kf} keyframes')
        print(prior.trainer.summary())

        # save() before release(), always: it goes through _ensure_live()
        trainer = prior.trainer
        extra = make_record(arm_out, split_at)(trainer, max(trainer.units - 1, 0))
        extra['label'] = label
        print(f'saved adapter to {prior.save(adapt_out, extra=extra)}')
        json.dump(trainer.log, open(f'{adapt_out}/{TRAIN_LOG}', 'w'))
        print(f'training log in {adapt_out}/{TRAIN_LOG}')
        # separate file, not a row in train_log.json: that log's record shape is a contract shared
        # with adapt/trainer.py, and a reader there would trip over a record with no 'loss'.
        # Written whenever the gate ran at all, INCLUDING the arrivals it let through - which is
        # what makes a threshold re-choosable without re-running.
        if trainer.gate_log:
            json.dump(trainer.gate_log, open(f'{adapt_out}/{GATE_LOG}', 'w'))
            print(f'gate log in {adapt_out}/{GATE_LOG}')
    finally:
        prior.release()          # in a finally: a crashed run otherwise strands ~2.5 GB
    free_vram('online adapt')

    res = evaluate(arm_out, label, split_at, e2e_cfg)
    print_report(res)
    free_vram('online eval')
    print(f'=== online adapt done in {time.time()-t0:.0f}s')
    return res
