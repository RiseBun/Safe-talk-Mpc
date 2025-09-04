# SafeTalk-MPC (PC 仿真 -> 实车迁移骨架)

## 环境
```bash
conda create -n mpcenv python=3.10 -y
conda activate mpcenv
conda install -c conda-forge casadi matplotlib numpy ffmpeg -y
```

## 运行
在 `sem2mpc` 目录下：
```bash
python -m sim.sim_runner dsl/example_task_curve_01.json
# 或：带自然语言指令
python -m sim.sim_runner dsl/example_task_curve_01.json "尽快到达目标，绕开障碍 0.4m"
```

输出：
- `mpc_result.png`
- `mpc_anim.mp4`
- `mpc_metrics.json`

## 批量实验（论文三件套）
```bash
python -m sim.batch_sweeps
```

## 目录
- `compiler/` 模型、构建 OCP、Shield（软/硬）
- `semantics/` LLM 规则回退、语义编译器
- `sim/` 仿真、动画、批量实验
