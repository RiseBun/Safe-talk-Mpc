# -*- coding: utf-8 -*-
import casadi as ca
from typing import Dict, Any, Tuple, List

from compiler.ackermann_model import AckermannModel
from compiler.load_task import load_task
from compiler.shield import soft_barrier


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _pad5(v: List[float]) -> List[float]:
    """把列表补到 5 维 [x,y,theta,v,delta]（不足补 0）"""
    v = list(v)
    if len(v) < 5:
        v = v + [0.0] * (5 - len(v))
    return v[:5]


def _expand_state_weights(w) -> List[float]:
    """
    输入可能是 3 维（x,y,theta）或 5 维（x,y,theta,v,delta）
    统一扩成 5 维：末两维默认 0.1
    """
    if w is None:
        return [10.0, 10.0, 1.0, 0.1, 0.1]
    w = list(w)
    if len(w) == 3:
        w = w + [0.1, 0.1]
    elif len(w) < 5:
        w = w + [0.1] * (5 - len(w))
    return w[:5]


def _expand_control_weights(r) -> List[float]:
    """
    控制权重允许是标量或向量；统一为 2 维 [a, delta]
    """
    if r is None:
        return [0.05, 0.05]
    if isinstance(r, (int, float)):
        return [float(r), float(r)]
    r = list(r)
    if len(r) == 1:
        return [float(r[0]), float(r[0])]
    return [float(r[0]), float(r[1])]


def apply_risk(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    风险自适应：调节安全半径/预测步长/终端权重/转角上限
    """
    risk = task.get('risk', 'med')
    obs = task.get('obstacle') or {}
    constraints = task.setdefault('constraints', {})

    if 'radius' not in obs:
        obs['radius'] = 0.35
    if 'center' not in obs:
        obs['center'] = [1.2, 0.5]
    task['obstacle'] = obs

    if 'horizon' not in task:
        task['horizon'] = 50
    if 'terminal_scale' not in task:
        task['terminal_scale'] = 3.0

    # 默认转角界
    dmax = float(constraints.get('delta_max', 0.5))
    dmin = float(constraints.get('delta_min', -dmax))

    if risk == 'low':
        obs['radius'] = float(obs.get('radius', 0.35)) - 0.05
        task['horizon'] = int(_clamp(task.get('horizon', 50) - 10, 20, 120))
    elif risk == 'high':
        obs['radius'] = float(obs.get('radius', 0.35)) + 0.15
        task['horizon'] = int(_clamp(task.get('horizon', 50) + 20, 20, 120))
        task['terminal_scale'] = float(task.get('terminal_scale', 3.0)) * 1.2
        # tighten steering
        dmax = _clamp(dmax * 0.8, 0.15, 0.6)
        dmin = -dmax

    constraints['delta_max'] = dmax
    constraints['delta_min'] = dmin
    task['constraints'] = constraints
    return task


def _apply_goal_bias_and_side(task: Dict[str, Any]) -> Tuple[List[float], List[float], str]:
    """
    读取 goal_bias 与 bias.side，返回 (x0, xf_biased, side)
    """
    x0 = _pad5(task['start'])
    xf = _pad5(task['goal'])

    # goal_bias: [dx, dy]
    bias = task.get('goal_bias', None)
    if bias and len(bias) >= 2:
        xf[0] += float(bias[0])
        xf[1] += float(bias[1])

    # bias.side: 'left' | 'right' | None
    side = None
    if isinstance(task.get('bias'), dict):
        side = str(task['bias'].get('side')).lower() if task['bias'].get('side') else None

    return x0, xf, side


def build_ocp(task_or_path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    构建 OCP：
      - 等式约束：初值 / 动力学离散化
      - 不等式约束：避障 (dist^2 - r^2 >= 0，硬约束)；控制上下界
      - 代价：状态（含终端）、控制、控制变化率、终端速度/转角、终端盒松弛
    返回：
      (nlp, meta) 其中 meta 包含 N, nx, nu, obstacle, bounds(lbg, ubg)
    """
    task = apply_risk(load_task(task_or_path))

    # 权重
    w_state = _expand_state_weights(task.get('weights', {}).get('state'))
    w_ctrl = _expand_control_weights(task.get('weights', {}).get('control'))
    terminal_scale = float(task.get('terminal_scale', 3.0))

    # 终端速度/转角权重（用于抑制过冲）
    q_vf = float(task.get('weights', {}).get('terminal_vel', 2.0))
    q_df = float(task.get('weights', {}).get('terminal_steer', 1.0))

    # 控制变化率权重（平滑控制；0 表示关闭）
    u_rate_w = float(task.get('u_rate_weight', 0.0))

    # 模型 & 时间栅格
    model = AckermannModel()     # X=[x,y,theta,v,delta], U=[a, delta_cmd]
    nx, nu = model.nx, model.nu

    N = int(task.get('horizon', 50))
    dt = float(task.get('dt', 0.1))
    dt = _clamp(dt, 0.02, 0.3)

    # 初末状态（并应用 goal 偏置 + 左/右侧偏中点）
    x0, xf, side = _apply_goal_bias_and_side(task)

    # diag 权重
    Q  = ca.diag(ca.DM(w_state))
    R  = ca.diag(ca.DM(w_ctrl))
    Qf = Q * terminal_scale

    # 变量
    X = ca.MX.sym('X', nx, N + 1)
    U = ca.MX.sym('U', nu, N)

    # 约束容器
    g_list: List[ca.MX] = []
    lbg: List[float] = []
    ubg: List[float] = []

    obj = 0

    # 初值等式：X0 = x0
    g_list.append(X[:, 0] - ca.DM(x0))
    lbg += [0.0] * nx
    ubg += [0.0] * nx

    # —— 中点引导（Method C），支持 left/right 绕行 —— #
    if task.get('insert_midpoint', True):
        mid = [(x0[0] + xf[0]) / 2.0, (x0[1] + xf[1]) / 2.0, 0.0, 0.0, 0.0]
        if side == 'left':
            mid[0] -= 0.30
            mid[1] += 0.20
        elif side == 'right':
            mid[0] += 0.30
            mid[1] -= 0.20
        via_points = [mid, xf]
    else:
        via_points = [xf]

    # 动力学 & 阶段代价
    for k in range(N):
        xk = X[:, k]
        uk = U[:, k]

        # 离散化（模型内部可为前向欧拉/更高阶）
        x_next = model.forward(xk, uk, dt)

        # 等式：X_{k+1} - f(X_k, U_k) = 0
        g_list.append(X[:, k + 1] - x_next)
        lbg += [0.0] * nx
        ubg += [0.0] * nx

        # 参考切换：前半段跟中点，后半段跟终点
        ref = via_points[0] if (len(via_points) > 1 and k < N // 2) else via_points[-1]
        ref = ca.DM(ref)

        # 阶段代价： (x-ref)^T Q (x-ref) + u^T R u
        obj += ca.mtimes([(xk - ref).T, Q, (xk - ref)]) + ca.mtimes([uk.T, R, uk])

        # 控制变化率正则： (u_k - u_{k-1})^2
        if u_rate_w > 0 and k >= 1:
            du = U[:, k] - U[:, k - 1]
            obj += u_rate_w * ca.mtimes(du.T, du)

    # —— 终端代价：位置 + 终端速度/转角抑制 —— #
    xN = X[:, -1]
    obj += ca.mtimes([(xN - ca.DM(xf)).T, Qf, (xN - ca.DM(xf))])
    obj += q_vf * (xN[3] ** 2) + q_df * (xN[4] ** 2)

    # —— 终端盒约束（带松弛，强制“到点且停住”趋势）—— #
    eps_pos = 0.05   # 终端盒尺寸（可调 0.05~0.1）
    w_box  = 50.0    # 终端盒松弛的权重
    w_vT   = 10.0    # 终端速度权重（补强）
    w_dT   = 5.0     # 终端转角权重（补强）

    sx = ca.MX.sym('sx')  # 终端 x 盒松弛
    sy = ca.MX.sym('sy')  # 终端 y 盒松弛

    # |x_N - x_f| <= eps + sx, |y_N - y_f| <= eps + sy
    g_list += [
        (X[0, -1] - xf[0]) - (eps_pos + sx),
        -(X[0, -1] - xf[0]) - (eps_pos + sx),
        (X[1, -1] - xf[1]) - (eps_pos + sy),
        -(X[1, -1] - xf[1]) - (eps_pos + sy),
    ]
    lbg += [-ca.inf, -ca.inf, -ca.inf, -ca.inf]
    ubg += [0.0, 0.0, 0.0, 0.0]

    # sx, sy >= 0
    g_list += [sx, sy]
    lbg += [0.0, 0.0]
    ubg += [ca.inf, ca.inf]

    # 终端附加代价（更强到点 + 停车）
    obj += w_box * (sx + sy) + w_vT * (X[3, -1] ** 2) + w_dT * (X[4, -1] ** 2)

    # —— 避障：硬/软/混合 —— #
    obs = task.get('obstacle', None)
    if obs:
        cx = float(obs['center'][0])
        cy = float(obs['center'][1])
        r = float(obs['radius'])
        r2 = r * r

        shield_cfg = task.get('shield', {}) or {}
        mode = str(shield_cfg.get('mode', 'hard')).lower()  # 'hard' | 'soft' | 'hybrid'
        soft_w = float(shield_cfg.get('weight', 8.0))

        use_hard = mode in ['hard', 'hybrid']
        use_soft = mode in ['soft', 'hybrid']

        for k in range(N + 1):
            dx = X[0, k] - cx
            dy = X[1, k] - cy
            dist2 = dx * dx + dy * dy
            if use_hard:
                # dist^2 - r^2 >= 0
                g_list.append(dist2 - r2)
                lbg.append(0.0)
                ubg.append(float('inf'))
            if use_soft:
                obj += soft_barrier(dist2, r2, w=soft_w)

    # —— 控制上下界：umin <= u <= umax —— #
    cons = task.get('constraints', {})
    a_min = float(cons.get('a_min', -1.0))
    a_max = float(cons.get('a_max',  +1.0))
    d_min = float(cons.get('delta_min', -0.5))
    d_max = float(cons.get('delta_max',  +0.5))

    umin = ca.DM([a_min, d_min])
    umax = ca.DM([a_max, d_max])
    for k in range(N):
        # u - umin >= 0
        g_list.append(U[:, k] - umin)
        lbg += [0.0] * nu
        ubg += [float('inf')] * nu
        # umax - u >= 0
        g_list.append(umax - U[:, k])
        lbg += [0.0] * nu
        ubg += [float('inf')] * nu

    # —— 打包 NLP（注意把 sx, sy 作为变量拼上）—— #
    vars_ = ca.vertcat(
        ca.reshape(X, -1, 1),
        ca.reshape(U, -1, 1),
        sx, sy
    )
    g = ca.vertcat(*g_list)
    nlp = {'x': vars_, 'f': obj, 'g': g}

    meta = {
        'N': N,
        'nx': nx,
        'nu': nu,
        'obstacle': obs,
        'bounds': {'lbg': lbg, 'ubg': ubg},
    }
    return nlp, meta
