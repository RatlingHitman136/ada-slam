"""~100-frame SLAM run to prove the 4x4 grid + new use_mono schedule actually execute."""
import os, sys
sys.path.insert(0,'/home/stud/treh/ada-slam'); os.chdir('/home/stud/treh/ada-slam')
import torch; torch.multiprocessing.set_start_method('spawn', force=True)
from adaslam.slam import SlamConfig, SlamRunner
from adaslam.runtime import ensure_venv_on_path, raise_fd_limit
raise_fd_limit(); ensure_venv_on_path()
cfg = SlamConfig(weights='pretrained_models/droid.pth', colors='data/KITTI/00/colors',
                 calib='data/KITTI/00/calib.txt', start=0, stop=100, undistort=False,
                 crop_border=0, stream_res=341*640, render_eval=False)
out = os.environ['CLAUDE_JOB_DIR'] + '/tmp/smoke_out'
r = SlamRunner(cfg).run(out, sys.argv[1], length=100, buffer=64)
import numpy as np
print(f'\nSMOKE OK  keyframes={len(np.loadtxt(out+"/traj_kf.txt"))}  '
      f'poses={len(np.loadtxt(out+"/traj_full.txt"))}')
