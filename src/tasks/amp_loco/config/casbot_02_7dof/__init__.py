"""Register Casbot 02 7DOF AMP locomotion tasks."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.amp_loco.rl import AMPOnPolicyRunner

from .env_cfgs import (
  casbot_02_7dof_amp_flat_env_cfg,
  casbot_02_7dof_amp_rough_env_cfg,
)
from .rl_cfg import casbot_02_7dof_amp_ppo_runner_cfg

# ── Rough terrain ──
register_mjlab_task(
    task_id="Casbot02-7DOF-AMP-Rough",
    env_cfg=casbot_02_7dof_amp_rough_env_cfg(),
    play_env_cfg=casbot_02_7dof_amp_rough_env_cfg(play=True),
    rl_cfg=casbot_02_7dof_amp_ppo_runner_cfg(),
    runner_cls=AMPOnPolicyRunner,
)

# ── Flat terrain ──
register_mjlab_task(
    task_id="Casbot02-7DOF-AMP-Flat",
    env_cfg=casbot_02_7dof_amp_flat_env_cfg(),
    play_env_cfg=casbot_02_7dof_amp_flat_env_cfg(play=True),
    rl_cfg=casbot_02_7dof_amp_ppo_runner_cfg(),
    runner_cls=AMPOnPolicyRunner,
)
