# 🛡️ SafeTalk-MPC: Semantic-to-MPC with Safety Shield

**SafeTalk-MPC** 是一个结合 **自然语言语义 (Semantic)** 与 **模型预测控制 (MPC)** 的框架。  
它支持用户直接用自然语言指令（如 *“绕障更保守，把安全半径改为 0.6 米”*）修改 **MPC 优化问题 (OCP)**，由 **语义编译器 (Semantic Compiler)** 自动转换为 JSON 补丁并应用到任务 DSL。  
同时，**安全盾 (Safety Shield)** 保证了轨迹的安全性与可行性，即使 LLM 生成的指令存在风险，也能自动兜底。

该项目既支持 **仿真环境验证**，也支持 **真实机器人（RDK X5 移动平台）实车部署**，为科研实验、论文发表和教学演示提供了开源基准。

---

## ✨ 创新点亮点

### 🔹 语义编译器 (Semantic Compiler)
- **功能**：自然语言 → DSL JSON Patch → 自动合成 MPC 问题。
- **支持修改**：目标点、避障半径、权重、预测步长、控制约束等。
- **创新**：实现了全自动语义到优化问题的转换，无需人工改配置。

### 🔹 安全盾 (Safety Shield)
- 三种模式：**硬约束 / 软约束 / 混合约束**。
- 当 LLM 生成不合理参数时，Shield 自动修正，保证安全优先。
- **创新**：语言驱动下的鲁棒 MPC 框架。

### 🔹 反回退约束 (No-Backtrack)
- 避免轨迹出现“回拉/倒退”。
- **创新**：强制单调前进，更符合直觉和真实机器人需求。

### 🔹 自适应限速 (Speed Cap)
- 远离目标时加速，接近目标时减速。
- **创新**：平稳逼近目标，避免过冲。

### 🔹 终端盒约束 (Terminal Box)
- 在目标点附近定义一个收敛盒。
- 要求进入此区域并低速停车。
- **创新**：保证“到点且停住”的稳定性。

### 🔹 风险自适应 (Risk-Adaptive)
- 根据 **风险等级 (low/med/high)** 自动调整安全半径、预测步长、转角范围。
- **创新**：语义层面对 MPC 配置的全局调优。

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

自动生成：
- `exp/base_result.png` (静态轨迹图)  
- `exp/base_anim.mp4` (轨迹动画)  
- `exp/base_metrics.json` (指标文件)  

### 3) 使用自然语言修改任务（语义编译器）
```powershell
# PowerShell 示例 (Here-String)
$instr = @'
绕障更保守，把安全半径加到 0.6 米
'@
python -m sim.sim_runner dsl/base.json $instr --llm ollama --model qwen2.5:7b --out exp/E1 --save-llm
```

将自动生成：
- `last_patch.json` → JSON 补丁  
- `_tmp_task.json` → 应用后的 DSL  
- 轨迹图与动画  

### 4) 使用离线补丁（无需 LLM）
```bash
echo "{ \"obstacle\": { \"radius\": 0.6 } }" > patch/I1.json
python -m sim.sim_runner dsl/base.json patch/I1.json --out exp/I1 --llm none
```

---

## 🧪 对照实验设计（验证语义编译器作用）

下表展示了自然语言指令与手工补丁的对应关系：

| 实验编号 | 自然语言指令 (LLM) | 手工补丁 JSON | 预期变化 |
|---|---|---|---|
| E1 | 绕障更保守，把安全半径改为 0.6 米 | `{"obstacle":{"radius":0.6}}` | min_obstacle_distance ↑ |
| E2 | 更强调平滑控制，把 u_rate_weight 提高到 1.0 | `{"u_rate_weight":1.0}` | 控制更平滑，end_err ↑ |
| E3 | 更靠近目标点停车，把 terminal_velocity 提到 60 | `{"weights":{"terminal_velocity":60}}` | 停车更稳，end_err ↓ |
| E4 | 缩短预测步长到 120 | `{"horizon":120}` | 求解更快 |
| E5 | 启用中点引导 | `{"insert_midpoint":true}` | 绕障路径更圆滑 |
| E6 | 靠近目标更慢：v_near=0.18, v_far=1.0 | `{"speed_cap":{"v_near":0.18,"v_far":1.0}}` | 更慢收敛，更平滑 |

**如何验证语义编译器生效？**
1. 运行自然语言指令 → 检查 `last_patch.json` 是否为正确的 JSON 补丁。  
2. 对比自然语言与手工补丁的结果 (`*_metrics.json` + 动画)。  
3. 若两者一致，说明语义编译器发挥了作用。  

---

## 📂 项目结构
```plaintext
SafeTalk-MPC/
├── sem2mpc/
│   ├── compiler/      # DSL 解析 & OCP 构建
│   ├── semantics/     # 语义编译器 & LLM Provider
│   ├── sim/           # 仿真与可视化
│   └── ...
├── dsl/               # 任务 DSL (JSON)
│   ├── base.json
│   └── example_task_curve_01.json
├── patch/             # JSON 补丁 (实验时用)
├── exp/               # 实验输出 (忽略追踪)
├── README.md
└── requirements.txt
```

---

## 🧹 .gitignore 建议
```gitignore
# 实验与临时文件
exp/
patch/
*_metrics.json
*_result.png
*_anim.mp4
last_patch.json
_tmp_task.json
llm_logs/

# Python 缓存
__pycache__/
*.pyc
*.pyo
*.pyd
```

---

## 📊 指标说明
- `end_position_error` → 末端误差（与目标点的距离，越小越好）  
- `min_obstacle_distance` → 最小障碍物距离（应 ≥ 安全半径）  
- `solve_time_sec` → 单次求解时间（可用于 horizon 对比）  

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
- 建立了 **自然语言 → JSON Patch → MPC 优化问题** 的全流程闭环。  
- 提出了 **安全盾** 与 **反回退 / 自适应限速 / 终端盒** 等一系列创新约束。  
- 提供了 **自然语言 vs 手工 JSON 补丁** 的可复现实验对照。  

这是一个既可用于 **科研（论文实验基准）**，也可用于 **教学和机器人应用** 的开源项目。  
