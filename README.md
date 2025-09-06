# 🛡️ SafeTalk-MPC: Semantic-to-MPC with Safety Shield

**SafeTalk-MPC** 是一个结合 **自然语言语义** 与 **模型预测控制（MPC）** 的框架。  
它支持用户用自然语言指令（例如 *“绕障更保守，把安全半径加到 0.6 米”*）直接修改 MPC 优化问题 (OCP)，并由 **安全盾 (Safety Shield)** 保证轨迹的可行性与安全性。  

本项目同时支持 **仿真验证** 与 **实车部署（如 RDK X5 移动机器人）**，提供科研实验基准和开源复现。

---

## ✨ 项目特点 & 创新点

- **语义编译器 (Semantic Compiler)**  
  自然语言 → DSL JSON Patch → 自动修改 MPC 参数（权重、约束、horizon、目标等）。

- **安全盾 (Safety Shield)**  
  支持 *hard / soft / hybrid* 多种避障策略；LLM 生成不合理参数时自动兜底，安全优先。

- **反回退约束 (No-Backtrack)**  
  抑制到点前的“回拉/倒退”。

- **自适应限速 (Speed Cap)**  
  远处更快、近端减速，避免过冲并提升收敛稳定性。

- **终端盒约束 (Terminal Box)**  
  约束末端进入收敛盒并低速停车，强化“到点且停住”。

- **风险自适应 (Risk-Adaptive)**  
  根据风险等级（low/med/high）自调安全半径、预测步长、转角限制，便于对比实验。

---

## 🚀 快速开始

### 1) 克隆仓库 & 安装依赖
```bash
git clone https://github.com/RiseBun/SafeTalk-MPC.git
cd SafeTalk-MPC/sem2mpc
conda create -n mpcenv python=3.10 -y
conda activate mpcenv
pip install -r requirements.txt
```

### 2) 运行基线实验
```bash
python -m sim.sim_runner dsl/example_task_curve_01.json --out exp/base --llm none
```

将生成：
- `exp/base_result.png`
- `exp/base_anim.mp4`
- `exp/base_metrics.json`

### 3) 使用**自然语言**修改任务（语义 → JSON Patch → MPC）
```powershell
# PowerShell 示例（Here-String 装下整段文字）
$instr = @'
绕障更保守，把安全半径加到 0.6 米
'@
python -m sim.sim_runner dsl/base.json $instr --llm ollama --model qwen2.5:7b --out exp/E1 --save-llm
```

将生成：
- `last_patch.json`（由 LLM 产出的 JSON Patch）
- `_tmp_task.json`（应用补丁后的 DSL）
- 对应的轨迹图与动画文件

### 4) 使用**离线补丁**（不依赖 LLM，最稳）
```bash
# 假设 dsl/base.json 已存在
echo "{ \"obstacle\": { \"radius\": 0.6 } }" > patch/I1.json
python -m sim.sim_runner dsl/base.json patch/I1.json --out exp/I1 --llm none
```

---

## 🧪 实验与复现（对照验证“语义编译器是否发挥作用”）

下表把**自然语言指令**与**手工 JSON 补丁**一一对应。若 LLM 生效，`last_patch.json` 应与手工补丁等价，且两者的 `metrics.json`、动画一致或接近。

| 编号 | 自然语言指令 (LLM) | 手工补丁 JSON | 预期变化 |
|---|---|---|---|
| E1 | 绕障更保守，把安全半径改为 0.6 米 | `{"obstacle":{"radius":0.6}}` | 最近障碍距离 ↑ |
| E2 | 更强调平滑控制，把 `u_rate_weight` 提高到 1.0 | `{"u_rate_weight":1.0}` | 控制更平滑，可能略增末端误差 |
| E3 | 更靠近目标点停车，把 `terminal_velocity` 罚权提到 60 | `{"weights":{"terminal_velocity":60}}` | 停车更稳，过冲风险↓ |
| E4 | 缩短预测步长到 120 | `{"horizon":120}` | 求解更快 |
| E5 | 启用中点引导，让路径更圆滑 | `{"insert_midpoint":true}` | 绕障姿态变化 |
| E6 | 靠近目标更慢：`v_near=0.18, v_far=1.0` | `{"speed_cap":{"v_near":0.18,"v_far":1.0}}` | 末端更慢、更平滑 |

**如何判定语义编译器生效？**
1. 运行带自然语言指令的命令会生成 `last_patch.json`；其内容应为**严格 JSON**。  
2. 对比 `last_patch.json` 与“手工补丁”是否语义等价；  
3. 对比两次运行的 `*_metrics.json` 与动画轨迹是否一致或接近（允许迭代/数值差异）。

---

## 📂 目录结构（建议）
```plaintext
SafeTalk-MPC/
├── sem2mpc/
│   ├── compiler/             # DSL 解析 & OCP 构建（Ackermann 模型、约束、权重等）
│   ├── semantics/            # 语义编译器 & Provider（Ollama / 多模型自一致性）
│   ├── sim/                  # 仿真与可视化（求解、绘图、动画、指标）
│   └── ...
├── dsl/                      # 任务 DSL (JSON)
│   ├── base.json
│   └── example_task_curve_01.json
├── patch/                    # JSON 补丁（建议忽略追踪）
├── exp/                      # 实验结果（建议忽略追踪）
├── README.md
└── requirements.txt
```

---

## 🧹 建议的 .gitignore 片段
```gitignore
# 结果与临时物
exp/
patch/*.tmp
*_metrics.json
*_result.png
*_anim.mp4
last_patch.json
_tmp_task.json
llm_logs/

# Python 常见
__pycache__/
*.pyc
*.pyo
*.pyd
.env
.venv
```

---

## 📊 指标说明（metrics.json 字段）

- `end_position_error`：最终位置误差（与目标点距离，越小越好）  
- `min_obstacle_distance`：最近障碍物距离（应 ≥ 安全半径）  
- `solve_time_sec`：单次求解时间（用于对比不同 horizon / 约束配置）

---

## 📖 引用
```bibtex
@misc{SafeTalkMPC2025,
  author = {RiseBun},
  title = {SafeTalk-MPC: Semantic-to-MPC with Safety Shield},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/RiseBun/SafeTalk-MPC}}
}
```

---

## 🏆 总结
- **自然语言 → JSON Patch → MPC 优化问题** 的全链路闭环；  
- **安全盾 + 反回退 + 自适应限速 + 终端盒** 等一组实用且可复现的约束；  
- 支持 **LLM 在线语义编译** 与 **离线 JSON 补丁** 双路径对照验证。

该项目既适合科研（论文实验基准），也适合教学和机器人应用的快速验证与演示。
