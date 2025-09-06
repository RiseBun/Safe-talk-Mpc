# 🛡️ SafeTalk-MPC: Semantic-to-MPC with Safety Shield

**SafeTalk-MPC** 是一个结合 **自然语言语义** 与 **模型预测控制（MPC）** 的框架。  
它支持用户用自然语言指令（例如 *“绕障更保守，把安全半径加到 0.6 米”*）直接修改 MPC 优化问题 (OCP)，并由 **安全盾 (Safety Shield)** 保证轨迹的可行性与安全性。  

本项目同时支持 **仿真验证** 与 **实车部署（如 RDK X5 移动机器人）**，提供科研实验基准和开源复现。

---

## ✨ 项目特点 & 创新点

### 🔹 语义编译器 (Semantic Compiler)
- 将用户的自然语言需求编译为 **DSL JSON Patch**，自动修改 MPC 参数、权重或约束。
- 支持多类修改：
  - 轨迹目标（goal）
  - 避障安全半径（obstacle radius）
  - 控制/状态权重
  - 预测步长 (horizon)  
- **创新**：自然语言 → JSON Patch → 自动合成 MPC 问题，全流程无人工干预。

### 🔹 安全盾 (Safety Shield)
- 支持 **硬约束 / 软约束 / 混合约束** 三种避障策略。
- **创新**：当 LLM 生成不合理的参数时，Shield 自动兜底，保证 **安全性优先**。

### 🔹 反回退约束 (No-Backtrack)
- 防止机器人在到达目标前出现“回拉/倒退”。
- **创新**：加入单调进展约束，使得轨迹更符合直觉，更适合真实机器人场景。

### 🔹 自适应限速 (Speed Cap)
- 根据与目标的距离动态调整速度：远处快、近处慢。
- **创新**：保证逼近目标时平稳收敛，避免过冲。

### 🔹 终端盒约束 (Terminal Box)
- 在目标点附近定义收敛盒，机器人必须进入此区域且低速停车。
- **创新**：提升“到点且停住”的可靠性。

### 🔹 风险自适应 (Risk-Adaptive)
- 用户可设定风险等级（low / med / high），自动调整安全半径、预测步长、转角范围。
- **创新**：不同风险等级下的自适应 MPC 配置，方便做对比实验。

---

## 🚀 快速开始

### 1. 克隆仓库 & 安装依赖
```bash
git clone https://github.com/RiseBun/SafeTalk-MPC.git
cd SafeTalk-MPC/sem2mpc
conda create -n mpcenv python=3.10
conda activate mpcenv
pip install -r requirements.txt
2. 运行基线实验
bash
复制代码
python -m sim.sim_runner dsl/example_task_curve_01.json --out exp/base --llm none
输出：

exp/base_result.png → 静态轨迹图

exp/base_anim.mp4 → 轨迹动画

exp/base_metrics.json → 指标（末端误差、最小障碍距离、求解时间等）

3. 使用自然语言修改任务
powershell
复制代码
# PowerShell 示例 (Here-String)
$instr = @'
绕障更保守，把安全半径加到 0.6 米
'@
python -m sim.sim_runner dsl/base.json $instr --llm ollama --model qwen2.5:7b --out exp/E1 --save-llm
会自动生成：

last_patch.json → JSON Patch

_tmp_task.json → 应用后的 DSL

轨迹图 & 动画

🧪 实验与复现
我们设计了 对照实验，验证语义编译器与手工 JSON Patch 一致：

编号	自然语言指令 (LLM)	手工补丁 JSON	预期变化
E1	“绕障更保守，把安全半径改为 0.6 米”	{"obstacle":{"radius":0.6}}	min_obstacle_distance ↑
E2	“更强调平滑控制，把 u_rate_weight 提高到 1.0”	{"u_rate_weight":1.0}	控制更平滑，end_err ↑
E3	“更靠近目标点停车，把 terminal_velocity 罚权提到 60”	{"weights":{"terminal_velocity":60}}	停车更稳，end_err ↓
E4	“缩短预测步长到 120”	{"horizon":120}	solve_time_sec ↓
E5	“启用中点引导，让路径更圆滑”	{"insert_midpoint":true}	避障姿态改变
E6	“靠近目标更慢：v_near=0.18，v_far=1.0”	{"speed_cap":{"v_near":0.18,"v_far":1.0}}	末端更慢，更平滑

通过对比 metrics.json（end_err / min_obstacle_distance / solve_time_sec）和 轨迹动画，可以直观看到自然语言控制的效果。

📂 目录结构
csharp
复制代码
SafeTalk-MPC/
├── sem2mpc/                  # 核心代码
│   ├── compiler/             # DSL 解析 & MPC 构建
│   ├── semantics/            # 语义编译器 & LLM Provider
│   ├── sim/                  # 仿真与可视化
│   └── ...
├── dsl/                      # 任务 DSL (JSON)
│   ├── base.json
│   └── example_task_curve_01.json
├── patch/                    # (忽略追踪) JSON 补丁
├── exp/                      # (忽略追踪) 实验结果
├── README.md                 # 工程说明
└── requirements.txt
.gitignore 已过滤 patch/, exp/, *.png, *.mp4, *_metrics.json 等实验结果，保证仓库整洁。

📊 指标说明
end_position_error → 最终位置误差（与目标点距离，越小越好）

min_obstacle_distance → 最近障碍物距离（需大于安全半径）

solve_time_sec → 单次求解时间（用于对比不同 horizon）

📖 引用
如果你在研究或发表论文中使用本项目，请引用：

bibtex
复制代码
@misc{SafeTalkMPC2025,
  author = {RiseBun},
  title = {SafeTalk-MPC: Semantic-to-MPC with Safety Shield},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/RiseBun/SafeTalk-MPC}}
}
🏆 总结
SafeTalk-MPC 实现了：

从 自然语言 → JSON Patch → MPC 优化问题 的全链路闭环；

安全盾 和 防回退 / 限速 / 终端盒 等一系列创新约束；

可复现、可对比的实验流程（自然语言 vs 手工补丁）。

这是一个既可用于科研（论文实验基准），也可用于教学和机器人应用的开源项目。
