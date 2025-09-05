# 🛡️ SafeTalk-MPC: Semantic-to-MPC with Safety Shield

**SafeTalk-MPC** 是一个将 **自然语言语义** 与 **模型预测控制（MPC）** 相结合的框架。  
它允许用户用自然语言指令（如 *“尽快到达目标，但保持 0.4m 安全距离，并更保守一些”*）直接调整优化问题 (OCP)，并由 **安全盾 (Safety Shield)** 保证可行性和安全性。  

本项目同时支持 **仿真验证** 与 **实车部署（如 RDK X5 移动机器人）**，提供论文实验基准和开源复现。

---

## ✨ 主要特性

- **语义编译器 (Semantic Compiler)**  
  - 将自然语言指令编译为 MPC 约束/权重/预测域的 JSON patch  
  - 内置 **单位检查 / 范围修正 / 冲突消解**  
  - 支持本地 LLM (Ollama) 或规则回退

- **安全盾 (Safety Shield)**  
  - 硬约束 (hard) + 软屏障 (soft) + 混合 (hybrid)  
  - 在 LLM 生成不安全/不可行参数时，自动修复或回退  
  - 确保轨迹 **永不越界**

- **风险自适应 (Risk-Adaptive MPC)**  
  - `risk=low|med|high` 自动调整：
    - 安全半径 `obstacle.radius`  
    - 预测步长 `N` 与步长 `dt`  
    - 终端权重 `terminal_scale`  
    - 转角约束 `delta_max/min`  

- **多时间尺度架构**  
  - 高层（语义/LLM） → 中层（安全盾） → 低层（实时 MPC）  
  - 支持任务切换与动态调权

- **实验友好**  
  - 每次运行保存：
    - `last_patch.json` → LLM 修改  
    - `_tmp_task.json` → 完整任务 DSL  
    - `*_metrics.json` → 指标  
    - `llm_runs.csv` → 实验日志  

---

## 🚀 快速开始

### 1. 环境配置
```bash
conda create -n mpcenv python=3.10 -y
conda activate mpcenv

pip install -r requirements.txt
2. 启动本地 LLM (Ollama)
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b-instruct
ollama serve

3. 基础运行
# 基线（无 LLM）
python -m sim.sim_runner dsl/example_task_curve_01.json --out mpc_base --llm none

4. 启用 LLM 语义编译
# 使用自然语言修改 OCP
python -m sim.sim_runner dsl/example_task_curve_01.json \
  "尽快到达，安全距离0.4m，更保守一些" \
  --llm ollama --model qwen2.5:7b-instruct --temp 0.0 --seed 42 \
  --save-llm --out mpc_llm

🧩 框架一图流
自然语言指令 + 场景上下文
        │
        ▼
  语义编译器 (semantics/)
        │
  └─ JSON Patch (last_patch.json)
        │
        ▼
  DSL 任务 (_tmp_task.json)
        │
        ▼
  构建 OCP (compiler/build_ocp.py)
        │
        ▼
  安全盾 (硬/软屏障 + 风险自适应)
        │
        ▼
  IPOPT 求解 (CasADi)
        │
        ▼
  可视化与指标 (sim/)
  ├─ 轨迹图 *.png, 动画 *.mp4
  └─ 指标 *.json + 日志 llm_runs.csv

🔑 关键模块

语义编译器（semantics/）

LLM 或规则回退生成 JSON Patch

Patch 应用到 DSL，得到 _tmp_task.json

OCP 构建（compiler/build_ocp.py）

动力学：AckermannModel

代价：状态/控制/终端速度/转角/控制变化率

约束：动力学一致性、输入界限、避障（硬/软）、终端盒

风险自适应：调节半径 / N / 终端权重 / 控制界

仿真与可视化（sim/）

sim_runner.py：一键运行

plot_animation.py：轨迹动画（或静态图兜底）

🎯 为什么与众不同？

语义→优化可验证：
LLM 不是直接控制，而是输出结构化参数修改，可解释、可验证。

安全盾护航：
确保所有 LLM 修改不破坏可行性。

实验可复现：
每次运行都保存 Patch/DSL/指标/日志，便于对比。

本地可运行：
支持 Ollama 本地模型 (qwen2.5:3b/7b-instruct)，无需云 API。

⚙️ LLM 可控的量

terminal_scale → 终端吸引力

weights.state/control → 代价权重

horizon, dt → 预测域

obstacle.radius → 安全半径

constraints.delta_max/min → 转角界限

goal_bias, bias.side=left|right → 轨迹绕行偏置

u_rate_weight → 控制平滑项

risk=low|med|high → 风险等级调节

🔍 如何验证 LLM 的作用？

观察以下文件/指标：

last_patch.json → LLM 修改了哪些参数？

_tmp_task.json → DSL 最终配置是否改变？

轨迹/指标 →

更安全？（最近障碍距离 ↑）

更稳？（过冲 ↓）

更快？（收敛时间 ↓）

🛑 内置“抗过冲/可行性”机制

终端盒 + 松弛变量：避免越过目标再折返

末端速度/转角惩罚：靠近目标自动减速

控制变化率惩罚：保证平顺

硬避障 + 软屏障：确保安全裕量

风险自适应：不同场景自动调节

左右绕显式偏置：消除原地犹豫

📊 实验与指标

参数灵敏度实验：terminal_scale, horizon

障碍位置扫描：碰撞率 vs 最近间距

控制界限收紧：可行率 vs 误差

LLM 边际作用：--llm none vs --llm ollama

所有结果记录在 llm_runs.csv。

🤖 面向实车部署（RDK X5）

接口：MPC 输出 (a, δ_cmd) 或 (v, ω)，可映射到 ROS 2 /cmd_vel

运行周期：推荐 50–100ms

感知输入：障碍可通过 JSON 动态更新

安全策略：必要时收紧转角/速度

📝 三句话总结

我们不是让 LLM 直接控制，而是让它把“人话”编译成可验证的 OCP；

安全盾保证 LLM 永不越界；

每次运行都可复现、可对比，清晰看到 LLM 的作用。
