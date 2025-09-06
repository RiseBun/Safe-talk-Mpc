# -*- coding: utf-8 -*-
import json
import re
from typing import Any, Dict, Tuple

from compiler.load_task import load_task, apply_patch
from compiler.shield import sanitize_patch  # 新增：安全裁剪

_JSON_KEY_RE = re.compile(r'([{\s,])([A-Za-z_][A-Za-z0-9_]*)\s*:')

def _extract_jsonish_block(text: str) -> str:
    """从任意文本中提取第一个 {...} 片段"""
    if not isinstance(text, str):
        raise TypeError(f"Expected string for patch text, got {type(text)}")
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
    把 LLM 输出（dict 或混合文本）转为严格 JSON 对象：
      - 先抽 {...}，尝试 json.loads
      - 失败则：单引号->双引号，未引号键补引号，再 loads
    """
    if isinstance(patch_any, dict):
        return patch_any
    if not isinstance(patch_any, str):
        raise TypeError(f"Patch must be dict or string, got {type(patch_any)}")

    s = _extract_jsonish_block(patch_any).strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        raise TypeError(f"Patch JSON must be an object, got {type(obj)}")
    except Exception:
        pass

    s2 = s.replace("'", '"')
    s2 = _JSON_KEY_RE.sub(r'\1"\2":', s2)
    obj2 = json.loads(s2)
    if not isinstance(obj2, dict):
        raise TypeError(f"Patch JSON must be an object, got {type(obj2)}")
    return obj2

def compile_from_text(task_or_path: Any, instruction: str, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    语义编译主入口（健壮版）：
      - instruction/llm_patch 里带 JSON 或“文本+JSON”皆可
      - 自动修复非严格 JSON；对补丁做 Safety Shield 裁剪
    返回：patched_task, patch_obj
    """
    patch_text_or_obj = kwargs.get("llm_patch", instruction)
    patch_obj = _coerce_to_json_obj(patch_text_or_obj)

    # Safety Shield：对补丁裁剪/保底（创新点的一部分）
    patch_obj = sanitize_patch(patch_obj)

    task_obj = load_task(task_or_path)
    patched = apply_patch(task_obj, patch_obj)
    return patched, patch_obj
