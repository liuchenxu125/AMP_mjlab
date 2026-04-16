# wbc_mjlab

本仓库用于 AMP 与速度任务相关的训练、脚本和资源管理。

## 当前上传范围

- rsl_rl
- scripts
- src
- setup.py

## mjlab 本地改动说明

你提到的符号 #sym:history_ordering 已在本机 mjlab 环境中生效。当前定位到的改动文件如下：

- /home/crp/miniconda3/envs/mjlab/lib/python3.11/site-packages/mjlab/managers/observation_manager.py

为便于迁移和复现，仓库中新增了补丁目录并保存该文件副本：

- mjlab_patch/mjlab/managers/observation_manager.py

主要改动点：

1. 在 ObservationGroupCfg 中增加了 history_ordering 配置项，可选 term 或 time。
2. 在观察项准备阶段，当 history_ordering=time 时，关闭每个 term 的历史维度扁平化，用于按时间步交错拼接。
3. 在拼接输出阶段，当结果是三维张量且 history_ordering=time 时，将结果 reshape 为二维输出。

对应代码位置（本机环境）：

- ObservationGroupCfg.history_ordering 定义：第 95 行附近
- 拼接后 time 布局处理：第 394 行附近
- term 配置阶段的 time 布局处理：第 452 行附近

说明：

- 以上 mjlab 改动位于 conda 环境 site-packages，不在本仓库版本管理内。
- 现在可直接使用仓库中的补丁副本重新覆盖目标环境文件。

示例（按当前 conda 环境路径）：

```bash
cp mjlab_patch/mjlab/managers/observation_manager.py \
	/home/crp/miniconda3/envs/mjlab/lib/python3.11/site-packages/mjlab/managers/observation_manager.py
```

## 本仓库中对该配置的使用

AMP 环境配置中，actor 与 critic 观测组已显式使用 history_ordering=time：

- src/tasks/amp_loco/amp_env_cfg.py

建议在复现实验时优先确认 mjlab 环境中的 observation_manager.py 与这里描述一致。
