# -*- coding: utf-8 -*-
import json
import re
from typing import Any, Dict, Tuple

from compiler.load_task import load_task, apply_patch

# 允许的“未加引号的 JSON 键名”：字母/数字/下划线开头
_JSON_KEY_RE = re.compile(r'([{\s,])([A-Za-z_][A-Za-z0-9_]*)\s*:')

def _extract_jsonish_block(text: str) -> str:
    """
    从任意文本中提取第一个 {...} 的片段（允许前后有自然语言）。
    若未找到，抛出 ValueError。
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected string for patch text, got {type(text)}")

    # 找到第一个 '{'，然后从该处起做括号配对
    try:
        start = text.index('{')
    except ValueError:
        raise ValueError("No '{' found in LLM output; cannot locate JSON object.")

    depth = 0
    end = None
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None:
        raise ValueError("Unbalanced braces in LLM output; cannot extract a JSON object.")

    return text[start:end+1]

def _coerce_to_json_obj(patch_any: Any) -> Dict[str, Any]:
    """
    把 LLM 输出（dict 或 混合文本中的 JSON 片段）转成严格 JSON 对象：
      1) 若是 dict，原样返回
      2) 若是 str：
         - 先抽取首个 {...} 片段
         - 先尝试严格 json.loads()
         - 失败则做“宽容修复”：单引号→双引号；未引号键名自动补引号；再 loads
    """
    if isinstance(patch_any, dict):
        return patch_any
    if not isinstance(patch_any, str):
        raise TypeError(f"Patch must be dict or string, got {type(patch_any)}")

    # 只取第一个 {...} 片段，避免前后自然语言干扰
    s = _extract_jsonish_block(patch_any).strip()

    # 1) 严格 JSON 尝试
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        raise TypeError(f"Patch JSON must be an object, got {type(obj)}")
    except Exception:
        pass

    # 2) 宽容修复：单引号替换、未引号键名补全
    s2 = s.replace("'", '"')
    s2 = _JSON_KEY_RE.sub(r'\1"\2":', s2)

    obj2 = json.loads(s2)  # 若仍失败，会抛出 JSONDecodeError
    if not isinstance(obj2, dict):
        raise TypeError(f"Patch JSON must be an object, got {type(obj2)}")
    return obj2

def compile_from_text(task_or_path: Any, instruction: str, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    语义编译主入口（健壮版）：
      - 调用方可把 LLM 返回的字符串放在 instruction，或通过 llm_patch=... 显式传入
      - 我们先解析/修复补丁，再把补丁应用到 task
    返回：patched_task, patch_obj
    """
    # 优先使用 llm_patch（如果 sim_runner 传了），否则就用 instruction
    patch_text_or_obj = kwargs.get("llm_patch", instruction)

    # 解析/修复为 JSON 对象
    patch_obj = _coerce_to_json_obj(patch_text_or_obj)

    # 把 task（可能是路径/JSON字串/字典）统一解析为对象
    task_obj = load_task(task_or_path)

    # 应用补丁
    patched = apply_patch(task_obj, patch_obj)
    return patched, patch_obj
