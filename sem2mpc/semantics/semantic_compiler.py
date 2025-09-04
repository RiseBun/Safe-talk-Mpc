# -*- coding: utf-8 -*-
from compiler.load_task import apply_patch
from semantics.llm_agent import LLMInterpreter

def compile_from_text(task_dict, instruction, context=None, provider=None):
    agent = LLMInterpreter(provider=provider)  # ✅ 支持注入
    patch = agent.parse(instruction, context or {})
    return apply_patch(task_dict, patch), patch
