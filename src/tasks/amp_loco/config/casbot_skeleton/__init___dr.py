"""Casbot Skeleton AMP tasks — Domain Randomization enhanced variants.

Register Casbot-Skeleton-AMP-Flat-DR and Casbot-Skeleton-AMP-Rough-DR.
Run alongside the base tasks; import this module to register.
"""

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.amp_loco.rl import AMPOnPolicyRunner

from .env_cfgs_dr import (
  casbot_amp_flat_env_cfg_dr,
  casbot_amp_rough_env_cfg_dr,
)
from .rl_cfg import casbot_amp_ppo_runner_cfg   # same PPO config

register_mjlab_task(
  task_id="Casbot-Skeleton-AMP-Rough-DR",
  env_cfg=casbot_amp_rough_env_cfg_dr(),
  play_env_cfg=casbot_amp_rough_env_cfg_dr(play=True),
  rl_cfg=casbot_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="Casbot-Skeleton-AMP-Flat-DR",
  env_cfg=casbot_amp_flat_env_cfg_dr(),
  play_env_cfg=casbot_amp_flat_env_cfg_dr(play=True),
  rl_cfg=casbot_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)
