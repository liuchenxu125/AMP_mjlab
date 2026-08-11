from mjlab.tasks.registry import register_mjlab_task

from src.tasks.amp_loco.rl import AMPOnPolicyRunner

from .env_cfgs import (
  casbot02_amp_flat_env_cfg,
  casbot02_amp_rough_env_cfg,
)
from .rl_cfg import casbot02_amp_ppo_runner_cfg

register_mjlab_task(
  task_id="Casbot02-AMP-Rough",
  env_cfg=casbot02_amp_rough_env_cfg(),
  play_env_cfg=casbot02_amp_rough_env_cfg(play=True),
  rl_cfg=casbot02_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="Casbot02-AMP-Flat",
  env_cfg=casbot02_amp_flat_env_cfg(),
  play_env_cfg=casbot02_amp_flat_env_cfg(play=True),
  rl_cfg=casbot02_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

from .leg_env_cfgs import (
  casbot02_leg_amp_flat_env_cfg,
  casbot02_leg_amp_rough_env_cfg,
)
from .leg_rl_cfg import casbot02_leg_amp_ppo_runner_cfg

register_mjlab_task(
  task_id="Casbot02-Leg-AMP-Flat",
  env_cfg=casbot02_leg_amp_flat_env_cfg(),
  play_env_cfg=casbot02_leg_amp_flat_env_cfg(play=True),
  rl_cfg=casbot02_leg_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="Casbot02-Leg-AMP-Rough",
  env_cfg=casbot02_leg_amp_rough_env_cfg(),
  play_env_cfg=casbot02_leg_amp_rough_env_cfg(play=True),
  rl_cfg=casbot02_leg_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

from .dual_env_cfgs import casbot02_dual_leg_amp_flat_env_cfg
from .leg_rl_cfg import casbot02_leg_amp_dual_ppo_runner_cfg

register_mjlab_task(
  task_id="Casbot02-Leg-AMP-Dual",
  env_cfg=casbot02_dual_leg_amp_flat_env_cfg(),
  play_env_cfg=casbot02_dual_leg_amp_flat_env_cfg(play=True),
  rl_cfg=casbot02_leg_amp_dual_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)