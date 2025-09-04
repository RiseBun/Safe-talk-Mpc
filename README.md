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
conda install -c conda-forge casadi matplotlib numpy ffmpeg -y