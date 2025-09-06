# -*- coding: utf-8 -*-
"""
SafeTalk-MPC - simulation runner

功能：
- 读取 DSL(JSON) -> 调用 build_ocp 构建 NLP
- 自动根据 nlp['x'] 的真实长度构造初值（兼容新增 slack 变量）
- 求解并绘图/动画，保存指标
- 两种补丁方式：
  A) --llm none：第二个位置参数为本地补丁（JSON 文件或内联 JSON）
  B) --llm ollama：第二个位置参数为自然语言/JSON，由语义编译器处理
"""

import os
import sys
import json
import time
import csv
import argparse
import platform
import inspect
import numpy as np
import casadi as ca

# ✅ 无界面环境也能出图
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from compiler.build_ocp import build_ocp
from sim.plot_animation import plot_trajectory_animation
from semantics.semantic_compiler import compile_from_text

# 可选 provider（单模型）
try:
    from semantics.providers.ollama_provider import make_ollama_provider as _mk_single_provider
except Exception:
    _mk_single_provider = None

# 可选 provider（多模型/自一致性）
try:
    from semantics.providers.multi_model_provider import make_multi_model_provider as _mk_multi_provider
except Exception:
    _mk_multi_provider = None


# ---- 修复：Provider 兼容包装（完整定义这两个函数） ----
import inspect

def _make_single_provider(model, base_url, temperature, num_predict, seed, save_llm):
    """
    兼容不同版本的 make_ollama_provider：
      新版：make_ollama_provider(model, base_url, temperature=..., num_predict=..., seed=..., save_dir=...)
      旧版：make_ollama_provider(model, base_url, debug=bool)
      最旧：make_ollama_provider(model, base_url)
    """
    if _mk_single_provider is None:
        return None

    sig = inspect.signature(_mk_single_provider)
    kwargs = {"model": model, "base_url": base_url}

    # 优先新版
    if "temperature" in sig.parameters:
        kwargs.update({
            "temperature": float(temperature),
            "num_predict": int(num_predict),
            "seed": int(seed),
        })
        if "save_dir" in sig.parameters and save_llm:
            kwargs["save_dir"] = "llm_logs"
        if "timeout" in sig.parameters:
            kwargs["timeout"] = 120
        if "max_retries" in sig.parameters:
            kwargs["max_retries"] = 2
        try:
            return _mk_single_provider(**kwargs)
        except TypeError:
            pass  # 回退

    # 旧版（可能带 debug）
    if "debug" in sig.parameters:
        try:
            return _mk_single_provider(model=model, base_url=base_url, debug=bool(save_llm))
        except TypeError:
            pass

    # 最旧：位置参数
    try:
        return _mk_single_provider(model=model, base_url=base_url)
    except TypeError:
        return _mk_single_provider(model, base_url)


def _make_multi_provider(models, base_url, k_samples, temperature, num_predict, seed, save_llm):
    """
    兼容 make_multi_model_provider：
      常见签名：make_multi_model_provider(models, base_url, k_samples, temperature, num_predict, seed, debug_dir=None)
    """
    if _mk_multi_provider is None:
        return None

    sig = inspect.signature(_mk_multi_provider)
    kwargs = {
        "models": models,
        "base_url": base_url,
        "k_samples": int(k_samples),
        "temperature": float(temperature),
        "num_predict": int(num_predict),
        "seed": int(seed),
    }
    if "debug_dir" in sig.parameters and save_llm:
        kwargs["debug_dir"] = "llm_logs"

    try:
        return _mk_multi_provider(**kwargs)
    except TypeError:
        try:
            # 最小必要参数回退
            return _mk_multi_provider(models, base_url, int(k_samples))
        except TypeError:
            return None
# ---- 以上为修复块 ----

# -------------------------
# 工具函数
# -------------------------
def _load_json(path: str):
    """读取 JSON（兼容 UTF-8 带 BOM）。"""
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def _save_json(obj, path: str):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:  # 写出无 BOM
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def _ensure_meta(nlp, meta):
    """
    兼容不同 build_ocp 实现：
    - 新：build_ocp -> (nlp, meta)，meta = {'N','nx','nu','bounds':{'lbg','ubg'}, 'obstacle':{...}}
    - 旧：build_ocp -> (nlp, N, nx, nu) —— 仅兜底
    """
    if isinstance(meta, dict) and 'N' in meta:
        if 'bounds' not in meta or 'lbg' not in meta['bounds'] or 'ubg' not in meta['bounds']:
            ng = int(nlp['g'].size1())
            meta.setdefault('bounds', {})
            meta['bounds']['lbg'] = [0.0] * ng
            meta['bounds']['ubg'] = [1e9] * ng
        return meta

    if isinstance(meta, (list, tuple)) and len(meta) == 3:
        N, nx, nu = meta
    else:
        try:
            N, nx, nu = meta
        except Exception:
            raise RuntimeError("build_ocp 返回格式不兼容：需要 (nlp, meta) 或 (nlp, N, nx, nu)")

    ng = int(nlp['g'].size1())
    return {
        'N': int(N),
        'nx': int(nx),
        'nu': int(nu),
        'bounds': {'lbg': [0.0] * ng, 'ubg': [1e9] * ng},
        'obstacle': None
    }

def _jsonable_patch(patch: dict) -> dict:
    """把 patch 里的 tuple 转 list，便于 JSON 保存。"""
    out = {}
    for k, v in (patch or {}).items():
        out[k] = list(v) if isinstance(v, tuple) else v
    return out

def _append_csv_log(csv_path: str, row: dict):
    header = list(row.keys())
    exists = os.path.isfile(csv_path)
    _ensure_dir(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)

def _looks_like_json_text(s: str) -> bool:
    """粗判是否是内联 JSON：以 { 或 [ 开头。"""
    if not isinstance(s, str):
        return False
    t = s.lstrip()
    return t.startswith("{") or t.startswith("[")

def _is_json_path(s: str) -> bool:
    """粗判是否像 JSON 文件路径：包含 / 或 \ 或 .json 结尾。"""
    if not isinstance(s, str):
        return False
    t = s.strip().strip('"').strip("'")
    return t.lower().endswith(".json") or ("/" in t) or ("\\" in t)

def _load_patch_from_arg(arg: str):
    """
    从第二个参数读取补丁：
      - 若是内联 JSON 文本：json.loads
      - 若是路径（相对/绝对）：打开读取（兼容 BOM）
      - 否则抛错（仅在 --llm none 下使用）
    """
    if _looks_like_json_text(arg):
        try:
            return json.loads(arg)
        except Exception as e:
            raise ValueError(f"Inline JSON patch is invalid: {e}")

    if _is_json_path(arg):
        p = arg.strip().strip('"').strip("'")
        if os.path.isfile(p):
            return _load_json(p)
        abs_p = os.path.abspath(p)
        if os.path.isfile(abs_p):
            return _load_json(abs_p)
        raise FileNotFoundError(f"Patch file not found: {arg} (abs: {abs_p})")

    raise ValueError("When --llm none, the second argument must be a JSON file path or an inline JSON object.")

def _deep_merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


# -------------------------
# 求解 + 作图
# -------------------------
def solve_and_plot(task_json_path, out_prefix='mpc'):
    """构建 NLP → 求解 → 画图/动画 → 指标保存；返回 metrics dict。"""
    print('🛠️ Building MPC problem from DSL...')
    build_ret = build_ocp(task_json_path)

    if isinstance(build_ret, (list, tuple)) and len(build_ret) == 2:
        nlp, meta = build_ret
    else:
        raise RuntimeError("build_ocp 必须返回 (nlp, meta)")

    meta = _ensure_meta(nlp, meta)
    N, nx, nu = meta['N'], meta['nx'], meta['nu']
    lbg, ubg = meta['bounds']['lbg'], meta['bounds']['ubg']

    # IPOPT 设置
    solver = ca.nlpsol('solver', 'ipopt', nlp, {
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.max_iter': 400,
        'ipopt.tol': 1e-6
    })

    # ====== 关键修复：根据 nlp['x'] 的真实长度来构造初值 ======
    n_dec = int(nlp['x'].size1())               # 决策变量总长度
    base_len = nx * (N + 1) + nu * N            # 只包含 X,U 时的长度
    extra = n_dec - base_len                    # 新增的变量个数（比如 sx, sy -> 2）

    if extra < 0:
        raise RuntimeError(f"Internal error: decision size smaller than X/U block. n_dec={n_dec}, base_len={base_len}")

    x_init = ca.DM.zeros((nx, N + 1))
    u_init = ca.DM.zeros((nu, N))
    init_guess = ca.vertcat(ca.reshape(x_init, -1, 1), ca.reshape(u_init, -1, 1))
    if extra > 0:
        init_guess = ca.vertcat(init_guess, ca.DM.zeros(extra, 1))
    # ====== 修复结束 ======

    print('🚀 Solving MPC...')
    t0 = time.time()
    sol = solver(x0=init_guess, lbg=lbg, ubg=ubg)
    t1 = time.time()

    if 'x' not in sol:
        raise RuntimeError("❌ IPOPT 未返回解向量 x")

    x_opt = ca.reshape(sol['x'][:nx * (N + 1)], nx, N + 1)  # (nx, N+1)
    xs = x_opt.T.full()                                    # (N+1, nx)
    xs_xy = xs[:, :2]

    # 指标
    task_cfg = _load_json(task_json_path)
    goal = np.array(task_cfg.get('goal', [2, 1, 0, 0, 0]))[:2]
    end_err = float(np.linalg.norm(xs_xy[-1] - goal))
    tot_time = t1 - t0

    # 最近距离（如有障碍）
    obstacle = None
    min_dist = None
    if meta.get('obstacle'):
        cx = float(meta['obstacle']['center'][0])
        cy = float(meta['obstacle']['center'][1])
        r = float(meta['obstacle']['radius'])
        d = np.sqrt((xs_xy[:, 0] - cx) ** 2 + (xs_xy[:, 1] - cy) ** 2)
        min_dist = float(np.min(d))
        obstacle = (cx, cy, r)

    # 画静态图
    plt.figure(figsize=(6, 6))
    plt.plot(xs_xy[:, 0], xs_xy[:, 1], 'b-o', ms=3, label='trajectory')
    plt.scatter([xs_xy[0, 0]], [xs_xy[0, 1]], c='g', s=60, label='start')
    plt.scatter([goal[0]], [goal[1]], c='r', s=60, label='goal')
    if obstacle:
        circ = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='gray', alpha=0.35, label='obstacle')
        plt.gca().add_patch(circ)
    plt.axis('equal'); plt.grid(True); plt.legend(); plt.title('SafeTalk-MPC Trajectory')
    fig_path = f"{out_prefix}_result.png"
    _ensure_dir(fig_path)
    plt.savefig(fig_path, dpi=150); plt.close()
    print(f"📷 saved {fig_path}")

    # 动画（轨迹点太少则跳过）
    if xs.shape[0] >= 3:
        try:
            anim_path = f"{out_prefix}_anim.mp4"
            plot_trajectory_animation(xs[:, :3], anim_path, obstacle=obstacle or (1.0, 0.5, 0.3))
            print(f"🎥 saved {anim_path}")
        except Exception as e:
            print(f"⚠️ 动画导出失败（忽略）：{e}")
    else:
        fallback = f"{out_prefix}_anim_static.png"
        plt.figure(figsize=(6, 6))
        plt.plot(xs_xy[:, 0], xs_xy[:, 1], 'b-o', ms=3)
        if obstacle:
            circ = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='gray', alpha=0.35)
            plt.gca().add_patch(circ)
        plt.axis('equal'); plt.grid(True); plt.title('Animation Fallback')
        _ensure_dir(fallback)
        plt.savefig(fallback, dpi=150); plt.close()
        print(f"🖼️ 轨迹点过少，导出静态替代图：{fallback}")

    # 写指标
    metrics = {
        'N': int(N),
        'end_position_error': end_err,
        'min_obstacle_distance': min_dist,
        'solve_time_sec': tot_time
    }
    _save_json(metrics, f"{out_prefix}_metrics.json")
    print("📑 metrics:", metrics)
    return metrics


# -------------------------
# CLI
# -------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(description="SafeTalk-MPC simulation runner")
    p.add_argument('task', nargs='?', default='dsl/example_task_curve_01.json',
                   help='DSL JSON path (default: dsl/example_task_curve_01.json)')
    p.add_argument('instruction', nargs='?', default=None,
                   help='Natural language instruction / JSON patch / patch file path')
    p.add_argument('--out', default='mpc', help='output file prefix (default: mpc)')

    # LLM 相关
    p.add_argument('--llm', default='ollama', choices=['none', 'ollama'],
                   help='use local LLM to compile semantics (default: ollama)')
    p.add_argument('--model', default='qwen2.5:3b', help='LLM model name (default: qwen2.5:3b)')
    p.add_argument('--models', default=None,
                   help='comma-separated model list (e.g., "qwen2.5:7b-instruct,llama3.1:8b-instruct")')
    p.add_argument('--k', type=int, default=1, help='samples per model for self-consistency (default: 1)')
    p.add_argument('--temp', type=float, default=0.0, help='LLM temperature (default: 0.0)')
    p.add_argument('--seed', type=int, default=42, help='sampling seed (default: 42)')
    p.add_argument('--base-url', default='http://127.0.0.1:11434', help='Ollama base URL')
    p.add_argument('--save-llm', action='store_true', help='save raw LLM logs to llm_logs/')
    p.add_argument('--risk', default='high', choices=['low', 'med', 'high'],
                   help='risk hint for semantics (default: high)')
    return p


def main():
    args = build_arg_parser().parse_args()
    task_path = args.task
    out_prefix = args.out
    instruction = args.instruction

    # ===== 分支 1：--llm none，本地补丁（文件/内联 JSON） =====
    if instruction is not None and args.llm == 'none':
        print(f"🗂️ Local patch mode (--llm none). Patch arg: {instruction}")
        # 1) 读取补丁对象
        patch_obj = _load_patch_from_arg(instruction)
        # 2) 读取原始任务
        if os.path.isfile(task_path):
            base_task = _load_json(task_path)
        else:
            # 允许把内联 JSON 当 task 传入
            base_task = json.loads(task_path)
        # 3) 合并
        patched_task = _deep_merge(json.loads(json.dumps(base_task)), patch_obj)
        # 4) 保存补丁和临时任务
        _save_json(_jsonable_patch(patch_obj), "last_patch.json")
        _save_json(patched_task, "_tmp_task.json")
        print("🧩 Local patch 已保存：last_patch.json")
        print("🧾 Patched DSL 已保存：_tmp_task.json")
        task_to_solve = "_tmp_task.json"

        metrics = solve_and_plot(task_to_solve, out_prefix=out_prefix)

    # ===== 分支 2：使用 LLM 语义编译（ollama） =====
    elif instruction is not None and args.llm != 'none':
        print(f"🗣️ Instruction: {instruction}")

        provider = None
        if args.models:
            if _mk_multi_provider is None:
                print("⚠️ 未找到多模型 provider，改用单模型或本地规则回退。")
            else:
                model_list = [m.strip() for m in args.models.split(",") if m.strip()]
                provider = _make_multi_provider(
                    models=model_list,
                    base_url=args.base_url,
                    k_samples=max(1, args.k),
                    temperature=args.temp,
                    num_predict=256,
                    seed=args.seed,
                    save_llm=args.save_llm
                )
        elif _mk_single_provider is not None:
            provider = _make_single_provider(
                model=args.model,
                base_url=args.base_url,
                temperature=args.temp,
                num_predict=256,
                seed=args.seed,
                save_llm=args.save_llm
            )

        original_task = _load_json(task_path) if os.path.isfile(task_path) else task_path

        try:
            patched_task, patch = compile_from_text(
                original_task, instruction,
                context={'risk_hint': args.risk},
                provider=provider
            )
        except Exception as e:
            print(f"⚠️ LLM 编译失败，改用本地规则回退：{e}")
            patched_task, patch = compile_from_text(
                original_task, instruction,
                context={'risk_hint': args.risk},
                provider=None
            )

        _save_json(_jsonable_patch(patch), "last_patch.json")
        _save_json(patched_task, "_tmp_task.json")
        print("🧩 LLM patch 已保存：last_patch.json")
        print("🧾 Patched DSL 已保存：_tmp_task.json")

        metrics = solve_and_plot("_tmp_task.json", out_prefix=out_prefix)

    # ===== 分支 3：无补丁，直接跑基线 =====
    else:
        print("🗣️ No instruction. Run base task.")
        metrics = solve_and_plot(task_path, out_prefix=out_prefix)

    # 记录一条实验日志
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "llm": args.llm,
        "models": (args.models or args.model) if instruction and args.llm != 'none' else "",
        "k": args.k,
        "temp": args.temp,
        "seed": args.seed,
        "instruction": instruction or "",
        "task": task_path,
        "end_err": metrics.get("end_position_error"),
        "min_dist": metrics.get("min_obstacle_distance"),
        "solve_time": metrics.get("solve_time_sec"),
        "N": metrics.get("N"),
        "machine": platform.platform(),
    }
    try:
        _append_csv_log("llm_runs.csv", row)
        print("🧾 appended log to llm_runs.csv")
    except Exception as e:
        print(f"⚠️ CSV 记录失败（忽略）：{e}")

    print('✅ Done.')


if __name__ == '__main__':
    main()
