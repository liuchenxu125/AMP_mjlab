# AMP_mjlab Conda 环境创建教程

这份教程用于给 `/home/casbot/workspace/AMP_mjlab` 创建可训练、可播放、可 sim2sim 的 conda 环境。

建议环境名：`amp_mjlab`

## 1. 创建 conda 环境

```bash
conda create -n amp_mjlab python=3.11 -y
conda activate amp_mjlab
python -m pip install -U pip setuptools wheel
```

## 2. 进入项目目录

```bash
cd /home/casbot/workspace/AMP_mjlab
```

## 3. 安装 PyTorch

当前项目已验证可用的环境是 CUDA 12.8 版本 PyTorch。

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

检查 CUDA 是否可用：

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

如果这里提示 NVIDIA driver 太旧，需要升级显卡驱动，或者安装和当前驱动匹配的 PyTorch CUDA 版本。

## 4. 安装基础依赖

```bash
pip install mjlab==1.2.0
pip install mujoco==3.9.0 warp-lang==1.14.0 scipy onnx onnxruntime tensorboard
pip install tyro rich pyyaml matplotlib trimesh viser wandb
pip install mujoco-python-viewer
```

## 5. 安装项目自带 rsl_rl

安装命令：

```bash
pip install -e rsl_rl
```

## 6. 安装 AMP_mjlab 项目本身

```bash
pip install -e .
```

## 7. 应用 mjlab patch

本项目的 observation history 依赖 `history_ordering`，需要覆盖 mjlab 的 `observation_manager.py`。

```bash
cp mjlab_patch/mjlab/managers/observation_manager.py \
  "$CONDA_PREFIX/lib/python3.11/site-packages/mjlab/managers/observation_manager.py"
```

## 8. 验证关键包

```bash
python - <<'PY'
import torch
import mujoco
import warp
import mjlab
import scipy
import onnxruntime
from rsl_rl.runners.amp_on_policy_runner import AmpOnPolicyRunner

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("mujoco:", mujoco.__version__)
print("warp:", warp.__version__)
print("scipy:", scipy.__version__)
print("onnxruntime:", onnxruntime.__version__)
print("mjlab:", mjlab.__file__)
print("AMP runner OK:", AmpOnPolicyRunner)
PY
```

如果能正常打印 `AMP runner OK`，说明 `rsl_rl` 没问题。

## 9. 查看 CASBOT02 任务是否注册成功

先确认环境能找到 CASBOT02 的 AMP 任务：

```bash
python scripts/list_envs.py --keyword Casbot02
```

正常应该能看到：

```text
Casbot02-AMP-Flat
Casbot02-AMP-Rough
```


## 10. 转换 CASBOT02 动作数据

如果 `casbot_data` 里有新的 `.data` 文件，需要先转成训练用 `.npz`：

```bash
python scripts/casbot02_data_to_npz.py \
  --input-dir casbot_data \
  --output-dir src/assets/motions/casbot02/amp/WalkandRun
```

转换后可以检查输出目录：

```bash
ls src/assets/motions/casbot02/amp/WalkandRun
```

训练时会自动读取这个目录下的 `.npz` 动作数据。

## 11. CASBOT02 正式训练

正式训练平地任务：

```bash
python scripts/train.py Casbot02-AMP-Flat --env.scene.num-envs=4096
```

## 12. CASBOT02 Play 验证

训练出 checkpoint 后，可以用 `scripts/play.py` 验证策略。

示例：

```bash
python scripts/play.py Casbot02-AMP-Flat \
  --checkpoint-file logs/rsl_rl/casbot02_amp_locomotion/<时间戳>/model_1000.pt \
  --num-envs 1 \
  --viewer viser \
  --no-terminations True
```

可以在浏览器打开：

```text
http://localhost:8080
```
## 14. CASBOT02 ONNX sim2sim 验证

指定某个 ONNX：

```bash
python scripts/sim2sim_casbot02_amp_onnx.py \
  logs/rsl_rl/casbot02_amp_locomotion/<时间戳>/export/Casbot02-AMP-Flat_model_8000.onnx
```


MuJoCo 窗口里的速度控制：

```text
↑ / ↓     调前进/后退速度 vx
← / →     调转向速度 yaw
Space     速度清零
```