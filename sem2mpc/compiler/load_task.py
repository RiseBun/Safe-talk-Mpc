import json, copy

DEFAULTS = {
    "start": [0.0, 0.0, 0.0, 0.0, 0.0],
    "goal":  [2.0, 1.0, 0.0, 0.0, 0.0],
    "horizon": 50,
    "dt": 0.1,
    "weights": {
        "state": [10, 10, 1, 0.1, 0.1],
        "control": [0.05, 0.05]
    },
    "terminal_scale": 3.0,
    "constraints": {
        "a_min": -1.0, "a_max": 1.0,
        "delta_min": -0.5, "delta_max": 0.5
    },
    "obstacle": {"center": [1.0, 0.5], "radius": 0.35},
    "risk": "med",
    "shield": {"mode": "hard", "weight": 10.0},
    "insert_midpoint": True
}

def deep_get(d, path):
    cur = d
    for p in path.split('.'):
        if p not in cur: return None
        cur = cur[p]
    return cur

def deep_set(d, path, value):
    cur = d
    parts = path.split('.')
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def load_task(task_path_or_dict):
    if isinstance(task_path_or_dict, dict):
        base = copy.deepcopy(DEFAULTS)
        merge(base, task_path_or_dict)
        return base
    with open(task_path_or_dict, "r", encoding="utf-8") as f:
        user = json.load(f)
    base = copy.deepcopy(DEFAULTS)
    merge(base, user)
    return base

def merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and k in dst and isinstance(dst[k], dict):
            merge(dst[k], v)
        else:
            dst[k] = v

def _as_float_list(x):
    # 把任意序列转成 float 列表
    return [float(v) for v in list(x)]

def apply_patch(task_dict, patch):
    """
    Apply simple patch ops to task (set/add/add_int/mul/mul_clip/mul_clip_vec2/sym_from/scale).
    关键修复：当原值是 list/tuple 时，mul/scale 等操作逐元素应用，避免出现
    “can't multiply sequence by non-int of type 'float'”。
    """
    task = copy.deepcopy(task_dict)

    for key, op in patch.items():
        if key.startswith('_'):   # 跳过元信息（如 _hits）
            continue

        orig = deep_get(task, key)

        # 统一处理 tuple/list 形式的操作指令
        if isinstance(op, (tuple, list)) and len(op) >= 1:
            tag = op[0]

            if tag == "set":
                deep_set(task, key, op[1])

            elif tag == "add":
                if isinstance(orig, (list, tuple)):
                    # 列表 + 标量：逐元素加
                    val = float(op[1])
                    deep_set(task, key, [float(v) + val for v in _as_float_list(orig)])
                else:
                    deep_set(task, key, float(orig or 0.0) + float(op[1]))

            elif tag == "add_int":
                deep_set(task, key, int(orig or 0) + int(op[1]))

            elif tag == "mul" or tag == "scale":
                factor = float(op[1])
                if isinstance(orig, (list, tuple)):
                    deep_set(task, key, [float(v) * factor for v in _as_float_list(orig)])
                else:
                    deep_set(task, key, float(orig or 1.0) * factor)

            elif tag == "mul_clip":
                # 针对标量：乘以 factor 后再裁剪到 [lo, hi]
                factor, lo, hi = float(op[1]), float(op[2]), float(op[3])
                if isinstance(orig, (list, tuple)):
                    vals = [max(lo, min(hi, float(v)*factor)) for v in _as_float_list(orig)]
                    deep_set(task, key, vals)
                else:
                    val = max(lo, min(hi, float(orig or 1.0) * factor))
                    deep_set(task, key, val)

            elif tag == "mul_clip_vec2":
                # 专门给长度为2的向量（如 control 权重）做裁剪乘法
                factor, lo, hi = float(op[1]), float(op[2]), float(op[3])
                vec = _as_float_list(orig) if isinstance(orig, (list, tuple)) else [float(orig or 1.0)]*2
                vec = [max(lo, min(hi, v*factor)) for v in vec]
                deep_set(task, key, vec)

            elif tag == "sym_from":
                other_val = deep_get(task, op[1])
                if isinstance(other_val, (list, tuple)):
                    deep_set(task, key, [-float(v) for v in _as_float_list(other_val)])
                else:
                    deep_set(task, key, -float(other_val))

            else:
                # 未知操作符，直接写入
                deep_set(task, key, op)

        else:
            # 非操作指令，当作“直接设定”
            deep_set(task, key, op)

    return task
