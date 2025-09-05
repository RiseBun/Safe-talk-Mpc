# 🛡️ SafeTalk-MPC: Semantic-to-MPC with Safety Shield

**SafeTalk-MPC** 是一个将 **自然语言语义** 与 **模型预测控制（MPC）** 相结合的框架。  
它允许用户用自然语言指令（如 *“尽快到达目标，但保持 0.4m 安全距离，并更保守一些”*）直接调整优化问题 (OCP)，并由 **安全盾 (Safety Shield)** 保证可行性和安全性。  

本项目同时支持 **仿真验证** 与 **实车部署（如 RDK X5 移动机器人）**，提供论文实验基准和开源复现。

---

## ✨ 主要特性

- **语义编译器 (Semantic Compiler)**  
  将自然语言指令编译为 MPC 约束/权重/预测域的 JSON patch，带有单位检查、范围修正和冲突消解。

- **安全盾 (Safety Shield)**  
  在 LLM 生成不安全/不可行参数时，自动回退或修复（基于 CBF 或可行性检查），保证轨迹始终安全。

- **风险自适应 (Risk-Adaptive MPC)**  
  根据场景风险等级（low/med/high），动态调节安全裕量 `d_safe`、预测步长 `N`、输入惩罚权重。

- **多时间尺度架构**  
  高层（语义/LLM） → 中层（安全盾） → 低层（实时 MPC），支持任务切换和参数自适应。

- **仿真 + 实车一体化**  
  代码在 **CasADi/IPOPT** 仿真中可直接运行，亦可迁移到机器人平台进行闭环验证。

---

## 🚀 快速开始

### 1. 环境配置
```bash
conda create -n mpcenv python=3.10 -y
conda activate mpcenv


# SafeTalk-MPC：把“人话”安全地编译成可证明的 MPC

**SafeTalk-MPC** 是一个把自然语言任务与场景描述，编译成**可验证的最优控制问题（OCP）**并由 **MPC** 安全执行的系统框架。  
它把 **LLM 的高层理解/偏好表达** 和 **MPC 的低层约束/最优性** 有机结合，通过一层**安全盾**（Safety Shield）确保**永不越界**、**始终可行**，同时支持本地 LLM（Ollama）离线运行。

---

## 1. 我们在解决什么问题？

- **人机沟通鸿沟**：用户说“更保守点”“绕障 0.4 米”“尽快但要稳”，但 MPC 需要权重/约束参数。  
- **安全与可行性**：LLM 输出可能不安全/不可行，需要 MPC 层保障。  
- **可复现与可解释**：需要看到 LLM 改了什么、为什么有效，并可复现。

---

## 2. 框架一图流

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

markdown
复制代码

---

## 3. 关键模块

- **语义编译器（`semantics/`）**  
  - LLM 或规则回退 → JSON Patch  
  - Patch 应用到 DSL，生成 `_tmp_task.json`  

- **OCP 构建（`compiler/`）**  
  - 动力学模型：AckermannModel  
  - 代价：状态/控制/末端速度/转角/控制变化率  
  - 约束：动力学、输入界、避障（硬+软）、末端盒  
  - 风险自适应：自动调安全半径、预测域、终端权重、转角界  

- **仿真与可视化（`sim/`）**  
  - `sim_runner.py`：一键运行，支持 LLM/多模型/温度/采样等参数  
  - `plot_animation.py`：轨迹动画，可降级为静态图  

---

## 4. 为什么与众不同？

1. **语义→优化可验证**：自然语言不是直接控制，而是编译成 OCP 的**结构化参数**，可解释、可验证。  
2. **安全盾护航**：硬约束 + 软屏障 + 风险自适应，保证**永不越界**。  
3. **实验友好**：每次运行保存 Patch、DSL、指标、日志，方便对比 `--llm none` vs `--llm ollama`。  
4. **离线运行**：支持 Ollama 本地模型 (`qwen2.5:3b/7b-instruct`)，无需云 API。

---

## 5. LLM 可控的量

- **终端目标吸引力**：`terminal_scale`  
- **代价权重**：`weights.state/control`  
- **预测域**：`horizon` / `dt`  
- **安全半径**：`obstacle.radius`  
- **转角界**：`constraints.delta_max/min`  
- **轨迹形态偏置**：`goal_bias`, `bias.side=left|right`  
- **平滑项**：`u_rate_weight`  
- **风险等级**：`risk=low|med|high`

所有修改会保存到 **`last_patch.json`** 与 **`_tmp_task.json`**。

---

## 6. 如何验证 LLM 的作用？

对照实验：

```bash
# 基线（无 LLM）
python -m sim.sim_runner dsl/example_task_curve_01.json --out mpc_base --llm none

# 启用本地 LLM
python -m sim.sim_runner dsl/example_task_curve_01.json \
  "尽快到达，安全距离0.4m，更保守一些" \
  --llm ollama --model qwen2.5:7b-instruct --temp 0.0 --seed 42 \
  --save-llm --out mpc_llm
观察三点：

last_patch.json：LLM 是否改了参数（半径↑、终端权重↑、转角界↓）。

_tmp_task.json：是否修改了预测域/安全半径/终端规则。

指标/轨迹：是否更安全（间距↑）、更稳（过冲↓）、更快（收敛时间↓）。

7. 内置“抗过冲/可行性”机制
终端盒 + 松弛变量：必须进入终端小盒，否则付费，避免越过再折返。

末端速度/转角惩罚：靠近目标自动减速。

控制变化率惩罚：控制平顺，减少抖动。

硬避障 + 软屏障：永不穿障，且有“安全边界层”。

风险自适应：高风险 → 半径↑、N↑、终端权重↑、转角界↓。

左右绕显式偏置：bias.side=left|right + 中点引导，避免原地犹豫。

8. 实验与指标
参数灵敏度：改变 terminal_scale, horizon

障碍物位置扫描：碰撞率、最近间距

控制界限收紧：可行率 vs 终点误差

LLM 边际作用：--llm none vs --llm ollama

运行结果自动记录在 llm_runs.csv。

9. 面向实车部署（RDK X5）
接口：MPC 输出 (a, δ_cmd) 或 (v, ω)，可映射到 ROS 2 /cmd_vel。

周期：推荐 50–100ms；超时可减少 N 或增加 dt。

感知：静态障碍可直接写 DSL，动态障碍更新 JSON 即可。

安全：必要时收紧 delta_max 与速度。

10. 三句话总结
我们不是让 LLM 控制，而是把“人话”编译成可验证的最优控制问题；

安全盾保证 LLM 永不越界；

每次运行都可复现、可对比，清晰看到 LLM 的作用。
