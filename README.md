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


SafeTalk-MPC：把“人话”安全地编译成可证明的 MPC

SafeTalk-MPC 是一个把自然语言任务与场景描述，编译成可验证的最优控制问题（OCP）并由 MPC 安全执行的系统框架。
它把 LLM 的高层理解/偏好表达 和 MPC 的低层约束/最优性 有机结合，通过一层安全盾（Safety Shield）确保永不越界、始终可行，同时支持本地 LLM（Ollama）离线运行。

1. 我们在解决什么问题？

人机沟通鸿沟：传统 MPC 需要工程师手工调代价权重、约束、预测域；用户的“更保守点”“绕障0.4米”“尽快但要稳”难以直接落到数学模型。

安全与可行性：即使 LLM 能“理解需求”，直接把它的输出用作控制目标也可能不可行或不安全。

可复现与可解释：学术/工程实践需要能“复盘”的轨迹、参数、约束与求解日志，清楚地看到 LLM 到底改了什么、为什么有效。

SafeTalk-MPC 的答案：

人话 → 语义 DSL（JSON） → 可验证 OCP（目标/约束/预测域） → 安全盾过滤/修复 → IPOPT 求解 → 轨迹与指标
全链路可记录、可重放、可对比。

2. 框架一图流（ASCII）
自然语言指令 + 场景上下文
        │
        ▼
  语义编译器 (semantics/)
  ├─ LLMInterpreter  (llm_agent.py)
  ├─ Provider(本地LLM/规则回退)
  └─ JSON Patch (last_patch.json)
        │
        ▼
  DSL 任务 (dsl/*.json)  ←  Patched DSL (_tmp_task.json，可重放)
        │
        ▼
  构建OCP (compiler/build_ocp.py)
  ├─ AckermannModel
  ├─ 代价：状态/控制/末端速度与转角/控制变化率
  ├─ 约束：动力学/输入界/避障(硬+软屏障)/末端盒
  └─ 风险自适应：horizon/安全半径/终端权重/转角上限
        │
        ▼
  安全盾 (compiler/shield.py)
  ├─ 硬约束：dist^2 - r^2 >= 0
  └─ 软屏障：soft_barrier(·) 形成安全边界层
        │
        ▼
  IPOPT 求解 (CasADi)
        │
        ▼
  可视化与指标 (sim/)
  ├─ 静态图 *.png, 动画 *.mp4
  └─ 指标 *.json + 运行日志 llm_runs.csv

3. 关键模块与职责

语义编译器（semantics/）

llm_agent.py：把“尽快到达、0.4m 绕障、轻柔一点”等自然语言解析为参数补丁（JSON Patch）；可接本地 LLM（Ollama）或规则回退。

semantic_compiler.py：把 Patch 施加到任务 DSL 上，产出 _tmp_task.json（可直接重放）。

providers/ollama_provider.py / providers/multi_model_provider.py：本地大模型推理，支持多模型自一致（K 采样）。

OCP 构建（compiler/）

ackermann_model.py：小车动力学（x, y, θ, v, δ；输入 a, δ_cmd）。

build_ocp.py：OCP 组装（目标/约束/预测域），含三件“抗过冲/可行性”利器：

末端盒 + 松弛变量（必须进一个小盒，超出就付费）；

末端速度/转角惩罚（靠近目标自动减速直达）；

控制变化率惩罚（避免抖振与过冲）。
同时内置硬避障 + 软屏障（混合 Shield）与风险自适应（high/med/low 自动调安全半径、N、终端权重、转角上限）。

shield.py：软屏障函数（可调权重，形成安全边界层）。

load_task.py：任务 DSL 加载与校验。

仿真与可视化（sim/）

sim_runner.py：一键运行；支持 --llm/--model/--models/--k/--temp/--save-llm 等。

plot_animation.py：轨迹动画/最近障碍距离标记；自动降级导出静态图。

4. 为什么它与众不同？

语义→优化可验证
我们不是“让 LLM 输出控制量”，而是把语言编译为 OCP 的结构化元素（权重、约束、预测域、终端规则），并提供类型/范围/可行性过滤与自动修复策略。
这让系统可解释、可证明、可复现。

安全盾护航
LLM 的建议必须先过硬约束 + 软屏障与风险自适应的把关；不安全/不可行时，自动收紧参数或保底执行；系统“永不越界”。

实证友好
每次运行都会落盘：

last_patch.json（LLM/规则到底改了什么）；

_tmp_task.json（可直接重放）；

*_metrics.json（到达误差、最近障碍距离、求解时间、N）；

llm_runs.csv（可一行看清模型/温度/种子/指标）。
对比 --llm none 与 --llm ollama，LLM 的“作用边际”一眼可见。

离线、低门槛
原生支持 Ollama 本地模型（如 qwen2.5:3b/7b-instruct），不依赖云 API；Windows/Conda 环境即可跑通。

5. 语义控制都能改哪些量？

一句话的用户偏好，最终会落到如下“可验证”的 OCP 参数：

终端目标吸引力：terminal_scale（更快→更大；更稳→适度减小）

代价权重：weights.state/control（更平稳→增大输入惩罚；更灵活→减小）

预测域：horizon/dt（危险↑→N↑ 或 dt↓，以更远视野/更细时距求稳）

安全半径：obstacle.radius（“绕开 0.4 m” → 半径外扩）

转角界：constraints.delta_max/min（“更保守” → 收紧转角）

轨迹形态偏置：goal_bias、bias.side=left|right（强制左/右绕）

平滑项：u_rate_weight（限制控制变化率，抑制抖动与过冲）

风险等级：risk=low|med|high（触发半径/N/终端权重/转角上限的整体联动）

所有修改都体现在 last_patch.json 与 _tmp_task.json 中，方便追因与复现。

6. “LLM 真的有用吗？”——如何一眼看出来

对照实验（同一 DSL，切换是否启用 LLM）：

# 基线：不启用 LLM（语义编译器走规则回退）
python -m sim.sim_runner dsl/example_task_curve_01.json --out mpc_base --llm none

# 启用本地 LLM
python -m sim.sim_runner dsl/example_task_curve_01.json \
  "尽快到达，安全距离0.4m，更保守一些" \
  --llm ollama --model qwen2.5:7b-instruct --temp 0.0 --seed 42 \
  --save-llm --out mpc_llm


观察三件事：

last_patch.json：LLM 是否提出了不同于规则回退的补丁（如同时增加 obstacle.radius、收紧 delta_max、增大 terminal_scale、指定 bias.side 等）。

_tmp_task.json：应用补丁后的 DSL 是否在预测域、安全半径、末端规则上有变化。

指标 & 轨迹：*_metrics.json 与 *_result.png/*_anim.mp4 是否反映**更安全（间距↑）、更稳（过冲↓）、更快（总代价/时间↓）**的权衡。

如果你的语句“力度不够”，LLM 和规则回退可能给出相似补丁。试着明确语义：
“强制右侧绕开 0.5m，更平稳 且 不要超过目标”。

7. 已内置的“抗过冲/可行性”机制

终端盒 + 松弛变量：要求末端位置进入一个小盒（|xN-xf|≤ε+sx），越界就付费（sx/sy），让轨迹更直接地停在目标，避免先越过再折返。

末端速度/转角惩罚：靠近目标时自动“刹车直达”。

控制变化率惩罚：平顺控制，减少抖动。

硬避障 + 软屏障（混合 Shield）：硬约束保证绝不穿障；软屏障形成“边界层”让优化器更容易找到绕行解。

风险自适应：危险 ↑ → radius↑、N↑、delta_max↓、terminal_scale↑。

左右绕显式偏置：bias.side=left|right + 中点引导，避免在障碍面前原地犹豫。

8. 你能从这套系统直接获得的论文点

方法：语义到优化的编译器 + 安全盾（硬/软混合）+ 风险自适应 + 多时间尺度。

可验证性：单位/范围/冲突过滤与自动修复；失败→修复→成功的闭环流程。

可复现：全量 patch/DSL/指标/日志落盘，支持 A/B/C 组学术对比。

可扩展：同一接口可平移到移动小车/机械臂/AGV；本地 LLM，边缘可用。

9. 典型实验与指标（已支持一键记录）

参数灵敏度：terminal_scale、weights.control、horizon
指标：终点误差、总代价、最近障碍距离、求解时间。

障碍物位置扫描：obstacle.center.x 逐步移动
指标：碰撞率（最小约束值<0 次数）、最近间距、轨迹长度。

控制界限收紧：delta_max ∈ {0.5, 0.3, 0.2}
指标：可行率、失败模式（恢复失败 vs 终点误差大）。

LLM 作用边际：--llm none vs --llm ollama
指标：到达率、碰撞率、过冲距离/次数、平滑度（控制变化率范数）。

所有运行都会把关键信息写入 llm_runs.csv，方便统一作图与统计检验。

10. 走向实车：从仿真到 RDK X5

接口：MPC 输出 (v, ω) 或 (a, δ_cmd)；ROS 2 /cmd_vel 或串口下发。

周期：在 X5 上建议 50–100 ms 控制周期；若 IPOPT 超时，先减小 N/增大 dt、再减小软屏障权重或简化代价。

感知：先静态障碍（DSL 配置）；再接激光/视觉，把障碍 center/radius 在线更新即可。

安全：保留硬约束，必要时进一步缩小 delta_max 与峰值速度。

11. 已知边界与建议

极端大半径/窄通道：硬约束可导致不可行；请组合软屏障与中点偏置，或允许临时放宽（松弛变量）。

多障碍/局部极值：推荐多中点引导或启发式初值；或增加 bias.side。

语言歧义：LLM 可能输出不确定 patch；开启 --save-llm 记录原始响应，必要时增加系统提示与结构化输出要求。

12. 三句“项目定位台词”

我们不是把 LLM 当控制器，而是把“人话”编译成可验证的最优控制问题；

安全盾让 LLM 永不越界：硬约束 + 软屏障 + 风险自适应 + 末端规则；

全链路可复现：每一次语义 → Patch → OCP → 轨迹与指标都能重放与对比。
conda install -c conda-forge casadi matplotlib numpy ffmpeg -y
