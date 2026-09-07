"""RL configuration for CASBOT02 lower-body-observation AMP task."""

import os
from dataclasses import asdict, dataclass
from typing import Any

from mjlab.rl import RslRlPpoAlgorithmCfg

from src.assets.robots import CASBOT02_LEG_AMP_BODY_NAMES
from src.tasks.amp_loco.config.g1.rl_cfg import g1_amp_ppo_runner_cfg


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
class Casbot02LegAmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO configuration exposing RSL-RL's symmetry extension."""

  symmetry_cfg: dict[str, Any] | None = None


def casbot02_leg_amp_ppo_runner_cfg():
  """Create RL runner configuration for CASBOT02 lower-body-observation AMP task."""
  cfg = g1_amp_ppo_runner_cfg()
  cfg.experiment_name = "casbot02_leg_amp_locomotion"
  cfg.amp_reward_coef = 0.2
  cfg.save_interval = 1000
  cfg.amp_motion_files = os.path.normpath(
    os.path.join(_MOTION_DATA_DIR, "WalkandRun")
  )
  cfg.min_normalized_std = [0.05] * 12
  cfg.amp_body_names = CASBOT02_LEG_AMP_BODY_NAMES
  cfg.amp_anchor_name = "torso"
  cfg.algorithm = Casbot02LegAmpPpoAlgorithmCfg(
    **asdict(cfg.algorithm),
    symmetry_cfg={
      "data_augmentation_func": (
        "src.tasks.amp_loco.config.casbot02.leg_symmetry:"
        "compute_symmetric_states"
      ),
      "use_data_augmentation": True,
      "use_mirror_loss": True,
      "mirror_loss_coeff": 0.1,
    },
  )
  return cfg
