"""Export CASBOT02 leg AMP checkpoints to ONNX with obs normalization baked in.

Usage:
  python scripts/export_checkpoints.py logs/rsl_rl/casbot02_leg_amp_locomotion/2026-08-10_09-42-44
"""

import argparse
import os
import sys
from pathlib import Path
from dataclasses import asdict

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


class AmpOnnxModel(nn.Module):
    """ONNX-exportable model: normalize obs → actor MLP → action."""

    def __init__(self, actor: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.policy = actor
        # Small epsilon to avoid div-by-zero
        self.register_buffer("_mean", mean.to(torch.float32).reshape(1, -1))
        self.register_buffer("_std", std.clamp(min=1e-6).to(torch.float32).reshape(1, -1))

    def forward(self, obs):
        normalized = (obs - self._mean) / self._std
        return self.policy(normalized)

    def get_dummy_inputs(self):
        return torch.randn(1, self._mean.shape[1])

    @property
    def input_names(self):
        return ["obs"]

    @property
    def output_names(self):
        return ["actions"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = REPO_ROOT / log_dir

    export_dir = log_dir / "export"
    export_dir.mkdir(exist_ok=True)

    # Import tasks to register them
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

    task_id = "Casbot02-Leg-AMP-Flat"
    env_cfg = load_env_cfg(task_id)
    agent_cfg = asdict(load_rl_cfg(task_id))
    runner_cls = load_runner_cls(task_id)

    # Create env once
    env_cfg.scene.num_envs = 1
    print("Creating env (CPU, 1 env)...")
    env = ManagerBasedRlEnv(env_cfg, "cpu")
    vec_env = RslRlVecEnvWrapper(env)
    print("Creating runner...")
    runner = runner_cls(vec_env, agent_cfg, str(log_dir), "cpu")

    checkpoints = sorted(log_dir.glob("model_*.pt"))
    print(f"Exporting {len(checkpoints)} checkpoints...\n")

    for ckpt in checkpoints:
        onnx_name = f"Casbot02-Leg-AMP-Flat_{ckpt.stem}.onnx"
        onnx_path = export_dir / onnx_name
        if onnx_path.exists():
            print(f"  [skip] {onnx_name}")
            continue

        try:
            # Load checkpoint
            runner.load(str(ckpt))
            ckpt_dict = torch.load(str(ckpt), map_location="cpu", weights_only=False)

            # Get actor model and normalize stats
            actor = runner.alg.policy.actor
            norm = ckpt_dict["obs_norm_state_dict"]
            mean = norm["_mean"].squeeze(0)
            std = norm["_std"].squeeze(0)

            # Build export model
            model = AmpOnnxModel(actor, mean, std)
            model.to("cpu")
            model.eval()

            # Export
            torch.onnx.export(
                model,
                model.get_dummy_inputs(),
                str(onnx_path),
                export_params=True,
                opset_version=18,
                verbose=False,
                input_names=model.input_names,
                output_names=model.output_names,
                dynamic_axes={},
                dynamo=False,
            )
            print(f"  [OK] {ckpt.name} -> {onnx_name}")
        except Exception as e:
            print(f"  [FAIL] {ckpt.name}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
