# -*- coding: utf-8 -*-
"""
SafeTalk-MPC - simulation runner

功能：
- 读取 DSL(JSON) -> 调用 build_ocp 构建 NLP
- 自动根据 nlp['x'] 的真实长度构造初值（兼容新增 slack 变量）
- 求解并绘图/动画，保存指标
- （可选）用 LLM 将自然语言指令编译为 DSL 增量修改（支持 ollama 单/多模型）
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
    from semantics.providers.ollama_provider import make_ollama_provider as _mk_single_provider  # 可能是新/旧版本
except Exception:
    _mk_single_provider = None

# 可选 provider（多模型/自一致性）
try:
    from semantics.providers.multi_model_provider import make_multi_model_provider as _mk_multi_provider
except Exception:
    _mk_multi_provider = None


# -------------------------
# 工具函数
# -------------------------
def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _save_json(obj, path):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _ensure_dir(p):
    d = os.path.dirname(p)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def _ensure_meta(nlp, meta):
    """
    兼容不同 build_ocp 实现：
    - 推荐：build_ocp -> (nlp, meta)，meta = {'N','nx','nu','bounds':{'lbg','ubg'}, 'obstacle':{...}}
    - 旧式：build_ocp -> (nlp, N, nx, nu) —— 仅兜底，不建议依赖
    """
    if isinstance(meta, dict) and 'N' in meta:
        if 'bounds' not in meta or 'lbg' not in meta['bounds'] or 'ubg' not in meta['bounds']:
            ng = int(nlp['g'].size1())
            meta.setdefault('bounds', {})
            # ⚠️ 兜底策略：假定构造的约束为 “g >= 0”
            meta['bounds']['lbg'] = [0.0] * ng
            meta['bounds']['ubg'] = [1e9] * ng
        return meta

    # 旧接口容错
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


# -------------------------
# Provider 兼容包装
# -------------------------
def _make_single_provider(model, base_url, temperature, num_predict, seed, save_llm):
    """
    兼容不同版本的 make_ollama_provider：
      新版：make_ollama_provider(model, base_url, temperature=..., num_predict=..., seed=..., save_dir=...)
      旧版：make_ollama_provider(model, base_url, debug=bool) / (model, base_url)
    """
    if _mk_single_provider is None:
        return None

    sig = inspect.signature(_mk_single_provider)
    kwargs = {"model": model, "base_url": base_url}

    # 优先尝试新版参数
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
            pass  # 回退到旧版

    # 旧版：可能只有 debug 参数
    if "debug" in sig.parameters:
        kwargs["debug"] = bool(save_llm)
        try:
            return _mk_single_provider(**kwargs)
        except TypeError:
            pass

    # 最旧：仅 (model, base_url)
    try:
        return _mk_single_provider(model=model, base_url=base_url)
    except TypeError:
        # 有些实现是位置参数
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
        # 退化为最小必要参数
        try:
            return _mk_multi_provider(models, base_url, int(k_samples))
        except TypeError:
            return None


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
    extra = n_dec - base_len                     # 新增的变量个数（比如 sx, sy -> 2）

    if extra < 0:
        raise RuntimeError(f"Internal error: decision size smaller than X/U block. n_dec={n_dec}, base_len={base_len}")

    # 先按老方式构造 X,U 的初值
    x_init = ca.DM.zeros((nx, N + 1))
    u_init = ca.DM.zeros((nu, N))
    init_guess = ca.vertcat(ca.reshape(x_init, -1, 1), ca.reshape(u_init, -1, 1))

    # 若有新增变量（如 sx, sy），再补 0
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
                   help='Natural language instruction to patch DSL')
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

    # 如传了自然语言，走语义→DSL patch
    if args.instruction is not None:
        print(f"🗣️ Instruction: {args.instruction}")

        provider = None
        if args.llm == 'ollama':
            if args.models:
                if _mk_multi_provider is None:
                    print("⚠️ 未找到多模型 provider，改用本地规则回退。")
                    provider = None
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
            else:
                if _mk_single_provider is None:
                    print("⚠️ 未找到单模型 provider，改用本地规则回退。")
                    provider = None
                else:
                    provider = _make_single_provider(
                        model=args.model,
                        base_url=args.base_url,
                        temperature=args.temp,
                        num_predict=256,
                        seed=args.seed,
                        save_llm=args.save_llm
                    )

        # 读取原始 DSL
        original_task = _load_json(task_path) if os.path.isfile(task_path) else task_path

        # 调用语义编译
        try:
            patched_task, patch = compile_from_text(
                original_task, args.instruction,
                context={'risk_hint': args.risk},
                provider=provider
            )
        except Exception as e:
            print(f"⚠️ LLM 编译失败，改用本地规则回退：{e}")
            patched_task, patch = compile_from_text(
                original_task, args.instruction,
                context={'risk_hint': args.risk},
                provider=None
            )

        # 保存 patch & 临时 DSL，方便复现实验
        _save_json(_jsonable_patch(patch), "last_patch.json")
        tmp_task = "_tmp_task.json"
        _save_json(patched_task, tmp_task)
        task_path = tmp_task
        print("🧩 LLM patch 已保存：last_patch.json")
        print(f"🧾 Patched DSL 已保存：{tmp_task}")

    # 正式求解 + 作图
    metrics = solve_and_plot(task_path, out_prefix=out_prefix)

    # 记录一条实验日志
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": (args.models or args.model) if args.instruction else "",
        "k": args.k,
        "temp": args.temp,
        "seed": args.seed,
        "instruction": args.instruction or "",
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
