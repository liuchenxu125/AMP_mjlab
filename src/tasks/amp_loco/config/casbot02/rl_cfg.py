"""RL configuration for CASBOT02 AMP locomotion task."""

import os
from dataclasses import asdict, dataclass
from typing import Any

from mjlab.rl import RslRlPpoAlgorithmCfg

from src.assets.robots import CASBOT02_23DOF_AMP_BODY_NAMES
from src.tasks.amp_loco.config.g1.rl_cfg import g1_amp_ppo_runner_cfg
from src.tasks.amp_loco.mdp.casbot02_symmetry import amp_symmetry_cfg


_MOTION_DATA_DIR = os.path.join(
  os.path.dirname(os.path.abspath(__file__)),
  os.pardir,
  os.pardir,
  os.pardir,
  os.pardir,
  os.pardir,
  "src",
  "assets",
  "motions",
  "casbot02",
  "amp",
)


@dataclass
class Casbot02AmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """AMPPPO config exposing RSL-RL's optional symmetry extension."""

  symmetry_cfg: dict[str, Any] | None = None


def _with_loco_symmetry(cfg) -> None:
  """Enable HANDOFF loco sagittal symmetry (data aug + mirror loss 0.1)."""
  cfg.algorithm = Casbot02AmpPpoAlgorithmCfg(
    **asdict(cfg.algorithm),
    symmetry_cfg=amp_symmetry_cfg(),
  )


def casbot02_amp_ppo_runner_cfg():
  """Create RL runner configuration for CASBOT02 AMP locomotion task."""
  cfg = g1_amp_ppo_runner_cfg()
  _with_loco_symmetry(cfg)
  cfg.experiment_name = "casbot02_amp_locomotion"
  cfg.save_interval = 1000
  cfg.amp_reward_coef = 0.1
  # cfg.amp_task_reward_lerp = 0.25
  cfg.amp_motion_files = os.path.normpath(
    os.path.join(_MOTION_DATA_DIR, "WalkandRun_TurnBoost_v1")
  )
  cfg.min_normalized_std = [0.05] * 23
  cfg.amp_body_names = CASBOT02_23DOF_AMP_BODY_NAMES
  cfg.amp_anchor_name = "torso"
  return cfg
