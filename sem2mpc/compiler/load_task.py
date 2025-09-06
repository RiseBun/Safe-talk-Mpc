# -*- coding: utf-8 -*-
import json
import os
from typing import Any, Dict, List

# -------------------------
# Helpers
# -------------------------

def _is_path(s: str) -> bool:
    if not isinstance(s, str):
        return False
    return any(sep in s for sep in ['/', '\\']) or s.endswith('.json')

def _load_json_from_path(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _try_parse_json_string(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception as e:
        raise TypeError(
            "Patch must be a JSON object. Got a plain string that is not valid JSON.\n"
            f"String snippet: {s[:120]!r}\n"
            "Hint: Ensure your LLM outputs *only* a JSON object, with no extra text."
        ) from e

def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst

# -------------------------
# Public merge APIs
# -------------------------

def merge(dst: Dict[str, Any], src: Any) -> Dict[str, Any]:
    """
    合并补丁：
      - src 是 dict：直接合并
      - src 是 str：若像路径→读文件；否则按 JSON 字符串解析
    """
    if isinstance(src, dict):
        return _deep_merge(dst, src)

    if isinstance(src, str):
        if _is_path(src) and os.path.exists(src):
            obj = _load_json_from_path(src)
            if not isinstance(obj, dict):
                raise TypeError(f"Patch file {src!r} must contain a JSON object, got {type(obj)}")
            return _deep_merge(dst, obj)
        obj = _try_parse_json_string(src)
        if not isinstance(obj, dict):
            raise TypeError(f"Patch must be a JSON object after parsing string, got {type(obj)}")
        return _deep_merge(dst, obj)

    raise TypeError(f"Patch must be a JSON object or JSON string or file path, got {type(src)}")

def apply_patch(task: Any, patch: Any) -> Dict[str, Any]:
    """
    兼容旧接口：task 可为 dict/路径/JSON字符串；patch 同 merge 支持的三类。
    返回新对象（不改原对象）。
    """
    base = load_task(task)  # 关键：先把 task 统一解析为 dict
    base_copy = json.loads(json.dumps(base))  # 深拷贝
    return merge(base_copy, patch)

def apply_patches(task: Any, patches: List[Any]) -> Dict[str, Any]:
    cur = load_task(task)
    cur = json.loads(json.dumps(cur))
    for p in patches:
        cur = merge(cur, p)
    return cur

# -------------------------
# Task loader
# -------------------------

def load_task(task_or_path: Any) -> Dict[str, Any]:
    """
    读任务 JSON 或对象；支持可选 'base' 合并：
      { "base": "path/to/base.json", ...overrides... }
    """
    if isinstance(task_or_path, dict):
        user = task_or_path
    elif isinstance(task_or_path, str):
        if _is_path(task_or_path) and os.path.exists(task_or_path):
            user = _load_json_from_path(task_or_path)
        else:
            user = _try_parse_json_string(task_or_path)
    else:
        raise TypeError(f"Task must be a dict, JSON string, or path, got {type(task_or_path)}")

    if not isinstance(user, dict):
        raise TypeError(f"Task must be a JSON object, got {type(user)}")

    base = user.get('base') or user.get('_base')
    if base:
        if isinstance(base, str):
            base_obj = _load_json_from_path(base)
            if not isinstance(base_obj, dict):
                raise TypeError(f"Base file {base!r} must contain a JSON object, got {type(base_obj)}")
        elif isinstance(base, dict):
            base_obj = base
        else:
            raise TypeError(f"'base' must be path or dict, got {type(base)}")

        merged = json.loads(json.dumps(base_obj))
        user_wo_base = dict(user)
        user_wo_base.pop('base', None)
        user_wo_base.pop('_base', None)
        return _deep_merge(merged, user_wo_base)

    return user
