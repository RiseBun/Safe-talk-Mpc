import casadi as ca
from compiler.ackermann_model import AckermannModel
from compiler.load_task import load_task
from compiler.shield import soft_barrier

def apply_risk(task):
    risk = task.get('risk', 'med')
    if risk == 'low':
        task['obstacle']['radius'] = task['obstacle'].get('radius', 0.35) - 0.05
        task['horizon'] = max(30, int(task.get('horizon',50) - 10))
    elif risk == 'high':
        task['obstacle']['radius'] = task['obstacle'].get('radius', 0.35) + 0.15
        task['horizon'] = min(80, int(task.get('horizon',50) + 20))
        task['terminal_scale'] = task.get('terminal_scale', 3.0) * 1.2
        # tighten steering
        task['constraints']['delta_max'] = max(0.15, task['constraints'].get('delta_max',0.5) * 0.8)
        task['constraints']['delta_min'] = -task['constraints']['delta_max']
    return task

def build_ocp(task_path):
    task = apply_risk(load_task(task_path))

    model = AckermannModel()
    N = int(task['horizon'])
    dt = float(task.get('dt', 0.1))
    x0 = task['start']
    xf = task['goal']

    nx, nu = model.nx, model.nu
    X = ca.MX.sym('X', nx, N+1)
    U = ca.MX.sym('U', nu, N)

    Q = ca.diag(ca.DM(task['weights']['state']))
    R = ca.diag(ca.DM(task['weights']['control']))
    Qf = Q * float(task.get('terminal_scale', 3.0))

    # mid-point (method C)
    via_points = [xf]
    if task.get('insert_midpoint', True):
        mid = [(x0[0] + xf[0]) / 2 + 0.5, (x0[1] + xf[1]) / 2 - 0.2, 0.0, 0.0, 0.0]
        via_points = [mid, xf]

    g_list, lbg, ubg = [], [], []
    obj = 0

    # initial condition
    g_list.append(X[:,0] - ca.DM(x0)); lbg += [0]*nx; ubg += [0]*nx

    # dynamics and stage cost
    for k in range(N):
        xk = X[:,k]; uk = U[:,k]
        x_next = model.forward(xk, uk, dt)
        g_list.append(X[:,k+1] - x_next); lbg += [0]*nx; ubg += [0]*nx

        ref = via_points[0] if k < N//2 else via_points[-1]
        ref = ca.DM(ref)
        obj += ca.mtimes([(xk-ref).T, Q, (xk-ref)]) + ca.mtimes([uk.T, R, uk])

    # terminal cost
    obj += ca.mtimes([(X[:,-1]-ca.DM(xf)).T, Qf, (X[:,-1]-ca.DM(xf))])

    # obstacle avoidance (hard >= 0) + optional soft barrier
    obs = task.get('obstacle', None)
    if obs:
        cx, cy = obs['center']; r = obs['radius']
        r2 = r*r
        soft_w = float(task.get('shield',{}).get('weight', 10.0))
        use_soft = task.get('shield',{}).get('mode','hard') in ['soft','hybrid']
        use_hard = task.get('shield',{}).get('mode','hard') in ['hard','hybrid']

        for k in range(N+1):
            dx = X[0,k] - cx; dy = X[1,k] - cy
            dist2 = dx*dx + dy*dy
            if use_hard:
                g_list.append(dist2 - r2); lbg.append(0.0); ubg.append(ca.inf)
            if use_soft:
                obj += soft_barrier(dist2, r2, w=soft_w)

    # control bounds: umin <= u <= umax  -> two inequalities
    umin = [task['constraints']['a_min'], task['constraints']['delta_min']]
    umax = [task['constraints']['a_max'], task['constraints']['delta_max']]
    umin = ca.DM(umin); umax = ca.DM(umax)
    for k in range(N):
        g_list.append(U[:,k] - umin); lbg += [0]*nu; ubg += [ca.inf]*nu
        g_list.append(umax - U[:,k]); lbg += [0]*nu; ubg += [ca.inf]*nu

    vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    g = ca.vertcat(*g_list)
    nlp = {'x': vars, 'f': obj, 'g': g}

    meta = {'N': N, 'nx': nx, 'nu': nu, 'obstacle': obs, 'bounds': {'lbg': lbg, 'ubg': ubg}}
    return nlp, meta
