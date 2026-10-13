"""Shared CASBOT02 gait-clock parameters for training and deployment."""

import math

import numpy as np

GAIT_PERIOD = 1.0
GAIT_OFFSET = 0.5
GAIT_STANCE_RATIO = 0.55
GAIT_STANDING_THRESHOLD = 0.1
GAIT_OBSERVATION_LAYOUT = "leg_gait_phase_v1"


def validate_gait_parameters(
  gait_period: float, gait_offset: float, standing_threshold: float
) -> None:
  if not math.isfinite(gait_period) or gait_period <= 0:
    raise ValueError("gait_period must be finite and positive")
  if not math.isfinite(gait_offset) or not 0 <= gait_offset < 1:
    raise ValueError("gait_offset must be in [0, 1)")
  if not math.isfinite(standing_threshold) or standing_threshold < 0:
    raise ValueError("standing_threshold must be finite and nonnegative")


def phase_features_numpy(
  elapsed_s: float,
  command: np.ndarray,
  *,
  gait_period: float = GAIT_PERIOD,
  gait_offset: float = GAIT_OFFSET,
  standing_threshold: float = GAIT_STANDING_THRESHOLD,
) -> np.ndarray:
  """Return [sin(L), cos(L), sin(R), cos(R)] from simulation/control time.

  The clock continues during standing; only the visible phases are zeroed.
  Reset elapsed_s when enabling/resetting the policy, not on command changes.
  """
  validate_gait_parameters(gait_period, gait_offset, standing_threshold)
  # Match the float32 command visible to the training policy at the stand boundary.
  command = np.asarray(command, dtype=np.float32)
  if command.shape != (3,):
    raise ValueError(f"Expected a three-component velocity command, got {command.shape}")
  if np.linalg.norm(command) < np.float32(standing_threshold):
    return np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
  left = (elapsed_s % gait_period) / gait_period
  right = (left + gait_offset) % 1.0
  angles = 2.0 * math.pi * np.array([left, right])
  return np.array(
    [np.sin(angles[0]), np.cos(angles[0]), np.sin(angles[1]), np.cos(angles[1])],
    dtype=np.float32,
  )
