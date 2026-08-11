"""Dual-AMP environment config: adds moving_mask observation group."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from src.tasks.amp_loco.config.casbot02.env_cfgs import (
  casbot02_amp_flat_env_cfg as _base_flat_env_cfg,
)
from src.tasks.amp_loco.config.casbot02.leg_env_cfgs import (
  _apply_leg_only_overrides,
)


def _add_moving_mask(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add moving_mask observation group for dual-AMP routing."""
  import src.tasks.amp_loco.mdp as amp_mdp
  cfg.observations["moving_mask"] = ObservationGroupCfg(
    terms={
      "mask": ObservationTermCfg(
        func=amp_mdp.moving_mask_from_command,
        params={"command_name": "twist", "yaw_threshold": 0.2},
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
    history_length=1,
  )


def casbot02_dual_leg_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Dual-AMP flat terrain config with moving_mask observation."""
  cfg = _apply_leg_only_overrides(_base_flat_env_cfg(play=play))
  _add_moving_mask(cfg)
  return cfg
