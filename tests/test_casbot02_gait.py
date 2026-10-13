"""Regression checks for CASBOT02's gait observation/deployment contract."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from scripts.sim2sim_casbot02_leg_amp_onnx import (
  gait_parameters_from_metadata, get_obs_frame, resolve_observation_layout,
)
from src.tasks.amp_loco.config.casbot02.env_cfgs import casbot02_amp_flat_env_cfg
from src.tasks.amp_loco.config.casbot02.leg_env_cfgs import casbot02_leg_amp_flat_env_cfg
from src.tasks.amp_loco.gait import GAIT_OBSERVATION_LAYOUT, phase_features_numpy
from src.tasks.amp_loco.mdp.casbot02_symmetry import mirror_amp_observation
from src.tasks.amp_loco.mdp.observations import loco_leg_phase, loco_phase_features
from src.tasks.amp_loco.mdp.rewards import gait_phase_contact


def fake_env(steps, commands, forces=None):
  commands = torch.as_tensor(commands, dtype=torch.float32)
  env = SimpleNamespace(
    episode_length_buf=torch.as_tensor(steps), step_dt=0.02, num_envs=len(steps),
    command_manager=SimpleNamespace(get_command=lambda name: commands),
  )
  if forces is not None:
    env.scene = {"feet": SimpleNamespace(data=SimpleNamespace(force=torch.as_tensor(forces)))}
  return env


class GaitTests(unittest.TestCase):
  def test_torch_numpy_match(self):
    steps = [0, 5, 13, 25, 33, 49, 50, 99, 101, 2000]
    for command in ([0, 0, 0], [0.05, 0, 0], [0.1, 0, 0], [0.06, 0.08, 0],
                    [0.06, 0, 0.08],
                    [0.3, 0, 0], [-0.3, 0, 0], [0, 0, 0.6]):
      env = fake_env(steps, [command] * len(steps))
      deployed = np.stack([phase_features_numpy(s * 0.02, np.array(command)) for s in steps])
      np.testing.assert_allclose(loco_phase_features(env).numpy(), deployed, atol=2e-6)
    phases = loco_leg_phase(fake_env([5, 30], [[0.3, 0, 0]] * 2))
    torch.testing.assert_close(phases, torch.tensor([[0.1, 0.6], [0.6, 0.1]]))

  def test_standing_and_command_transitions(self):
    commands = [[0, 0, 0], [0.3, 0, 0], [0, 0, 0], [0, 0, -0.6]]
    features = loco_phase_features(fake_env([13] * 4, commands))
    torch.testing.assert_close(features[0], torch.tensor([0., 1., 0., 1.]))
    torch.testing.assert_close(features[0], features[2])
    torch.testing.assert_close(features[1], features[3])

  def test_contact_reward(self):
    forces = torch.zeros(6, 2, 3)
    forces[0, 0, 2] = 100
    forces[1, 1, 2] = 100
    forces[2, 1, 2] = 100
    forces[3, :, 2] = 100
    forces[4, :, 2] = 100
    forces[5, 0, 2] = 100
    env = fake_env([5, 5, 30, 5, 5, 5], [[0.3, 0, 0]] * 4 + [[0, 0, 0]] * 2, forces)
    expected = torch.tensor([2., 0., 2., 1., 2., 1.])
    torch.testing.assert_close(gait_phase_contact(env, "feet"), expected)
    env.scene["feet"].data.force = forces.unsqueeze(2)
    torch.testing.assert_close(gait_phase_contact(env, "feet"), expected)
    env.scene["feet"].data.force = None
    with self.assertRaises(ValueError):
      gait_phase_contact(env, "feet")

  def test_invalid_parameters(self):
    env = fake_env([0], [[0.3, 0, 0]], torch.zeros(1, 2, 3))
    for period in (0, -1, float("nan")):
      with self.assertRaises(ValueError):
        loco_phase_features(env, gait_period=period)
      with self.assertRaises(ValueError):
        phase_features_numpy(0, np.array([0.3, 0, 0]), gait_period=period)
    with self.assertRaises(ValueError):
      gait_phase_contact(env, "feet", stance_ratio=0)

  def test_mirror(self):
    for width in (180, 444, 196, 460, 308, 788):
      for ordering in ("time", "term"):
        obs = torch.randn(3, width)
        mirrored = mirror_amp_observation(obs, history_ordering=ordering)
        torch.testing.assert_close(
          mirror_amp_observation(mirrored, history_ordering=ordering), obs
        )
    for width in (196, 460):
      obs = torch.zeros(2, width).reshape(2, 4, width // 4)
      obs[..., 45:49] = torch.tensor([1., 2., 3., 4.])
      mirrored = mirror_amp_observation(obs.flatten(1)).reshape_as(obs)
      torch.testing.assert_close(mirrored[..., 45:49], obs[..., [47, 48, 45, 46]])

  def test_config_keeps_existing_rewards_and_amp(self):
    base = casbot02_amp_flat_env_cfg()
    cfg = casbot02_leg_amp_flat_env_cfg()
    self.assertEqual(set(cfg.rewards) - set(base.rewards), {"gait_phase_contact"})
    self.assertEqual(cfg.rewards["gait_phase_contact"].weight, 0.5)
    for name, term in base.rewards.items():
      self.assertEqual(cfg.rewards[name], term)
    self.assertEqual(list(cfg.observations["actor"].terms),
                     ["base_ang_vel", "projected_gravity", "command", "joint_pos",
                      "joint_vel", "actions", "phase"])
    self.assertEqual(list(cfg.observations["critic"].terms)[:7],
                     list(cfg.observations["actor"].terms))
    self.assertNotIn("phase", cfg.observations["amp"].terms)
    self.assertEqual(cfg.observations["actor"].history_length, 4)

  def test_sim2sim_layouts(self):
    data = SimpleNamespace(
      sensor=lambda name: SimpleNamespace(data=np.zeros(3)),
      qpos=np.r_[np.zeros(3), [1., 0., 0., 0.], np.zeros(22)], qvel=np.zeros(28),
    )
    args = (data, np.array([0.3, 0, 0]), np.arange(12), np.zeros(12))
    legacy = get_obs_frame(*args)
    current = get_obs_frame(*args, include_phase=True, elapsed_s=0.26)
    self.assertEqual(legacy.shape, (45,))
    self.assertEqual(current.shape, (49,))
    np.testing.assert_array_equal(current[:45], legacy)
    np.testing.assert_allclose(current[45:], phase_features_numpy(0.26, args[1]))
    self.assertEqual(resolve_observation_layout(180), (45, 4, False))
    self.assertEqual(resolve_observation_layout(196), (49, 4, True))
    for width in (None, 188, 308):
      with self.assertRaises(ValueError):
        resolve_observation_layout(width)

  def test_onnx_metadata(self):
    metadata = {
      "casbot02_observation_layout": GAIT_OBSERVATION_LAYOUT,
      "history_length": "4", "history_ordering": "time",
      "gait_phase_feature_order": "sin_left,cos_left,sin_right,cos_right",
      "gait_period": "0.8", "gait_offset": "0.5", "gait_standing_threshold": "0.1",
    }
    self.assertEqual(gait_parameters_from_metadata(metadata)["gait_period"], 0.8)
    for change in ({"gait_period": "0"}, {"history_ordering": "term"},
                   {"casbot02_observation_layout": "unknown"}):
      with self.assertRaises(ValueError):
        gait_parameters_from_metadata({**metadata, **change})
    with self.assertRaises(ValueError):
      gait_parameters_from_metadata({})


if __name__ == "__main__":
  unittest.main()
