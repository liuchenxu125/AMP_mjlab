"""CPU unit tests for the signed root-frame feet-crossing cost."""
from types import SimpleNamespace
import unittest

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply
from src.tasks.amp_loco.mdp.rewards import feet_crossing


def make_env(widths, quat=None, translation=None, stride=0.0):
  widths = torch.tensor(widths, dtype=torch.float32)
  n = len(widths)
  feet = torch.zeros(n, 2, 3)
  feet[:, 0, 1] = widths/2
  feet[:, 1, 1] = -widths/2
  feet[:, 0, 0] = stride/2
  feet[:, 1, 0] = -stride/2
  if quat is None:
    quat = torch.tensor([1., 0., 0., 0.]).repeat(n, 1)
  else:
    quat = torch.tensor(quat, dtype=torch.float32).repeat(n, 1)
    quat /= torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
  feet = quat_apply(quat[:, None, :].expand(-1, 2, -1), feet)
  if translation is not None:
    feet += torch.tensor(translation)
  data = SimpleNamespace(body_link_pos_w=feet, site_pos_w=feet.clone(), root_link_quat_w=quat)
  return SimpleNamespace(scene={"robot": SimpleNamespace(data=data)}, extras={"log": {}})


class FeetCrossingTests(unittest.TestCase):
  def setUp(self):
    self.cfg = SceneEntityCfg("robot", body_names=("left", "right"), body_ids=[0, 1], preserve_order=True)

  def test_hinge_and_signed_crossing(self):
    env = make_env([.285, .20, .18, .10, .0, -.10])
    result = feet_crossing(env, self.cfg, .20)
    torch.testing.assert_close(result, torch.tensor([0., 0., .02, .10, .20, .30]))
    self.assertAlmostEqual(env.extras["log"]["Metrics/feet_crossed_fraction"].item(), 1/6)
    self.assertAlmostEqual(env.extras["log"]["Metrics/feet_too_close_fraction"].item(), 4/6)

  def test_heading_translation_and_stride_invariance(self):
    base = feet_crossing(make_env([.18, -.10]), self.cfg, .20)
    # Nontrivial full root rotation, a large world translation and a long stride.
    moved = make_env([.18, -.10], quat=[.4, .2, -.3, .5], translation=[100., -30., 2.], stride=1.)
    torch.testing.assert_close(feet_crossing(moved, self.cfg, .20), base, atol=1.e-5, rtol=1.e-5)

  def test_sites(self):
    env = make_env([.18])
    cfg = SceneEntityCfg("robot", site_names=("left", "right"), site_ids=[0, 1], preserve_order=True)
    torch.testing.assert_close(feet_crossing(env, cfg, .20), torch.tensor([.02]))

  def test_mirror_symmetry(self):
    env = make_env([.18, -.10])
    original = feet_crossing(env, self.cfg, .20)
    mirrored = env.scene["robot"].data.body_link_pos_w[:, [1, 0]].clone()
    mirrored[..., 1] *= -1
    env.scene["robot"].data.body_link_pos_w = mirrored
    torch.testing.assert_close(feet_crossing(env, self.cfg, .20), original)

  def test_gradient_opposes_inward_motion(self):
    env = make_env([.18])
    pos = env.scene["robot"].data.body_link_pos_w.requires_grad_()
    feet_crossing(env, self.cfg, .20).sum().backward()
    torch.testing.assert_close(pos.grad[0, :, 1], torch.tensor([-1., 1.]))

  def test_invalid_selection_and_threshold(self):
    env = make_env([.18])
    for threshold in (0., -.1):
      with self.assertRaises(ValueError):
        feet_crossing(env, self.cfg, threshold)
    cfg = SceneEntityCfg("robot", body_names=("left",), body_ids=[0])
    with self.assertRaises(ValueError):
      feet_crossing(env, cfg, .20)


if __name__ == "__main__":
  unittest.main()
