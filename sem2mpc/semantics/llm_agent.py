# -*- coding: utf-8 -*-
import re, math, json

def _to_rad(val, unit):
    unit = (unit or "").lower()
    if unit in ["deg","degree","degrees","度"]:
        return float(val) * math.pi / 180.0
    return float(val)

class LLMInterpreter:
    """
    把 (自然语言, 场景上下文) -> 对 DSL 的“增量修改” patch (dict)。
    provider: 可注入函数 f(prompt:str)->str，返回 JSON 字符串；不提供则用本地规则回退。
    """

    def __init__(self, provider=None):
        self.provider = provider

    def _fallback_rules(self, text, context):
        patch = {"_hits": []}
        tl = text.lower()

        if re.search(r"(尽快|更快|asap|quick)", tl):
            patch["terminal_scale"] = ("scale", 1.5)
            patch["weights.control"] = ("mul", 0.7)
            patch["_hits"].append("speed_up")

        if re.search(r"(更平稳|舒适|smooth|comfort)", tl):
            patch["weights.control"] = ("mul", 1.5)
            patch["_hits"].append("smoother")

        m = re.search(r"(安全距离|safe distance).*?([0-9.]+)\s*(m|米)", tl)
        if m:
            val = float(m.group(2))
            # 把用户说的“绕开 0.x m”转成障碍半径增加（保守一点）
            patch["obstacle.radius"] = ("add", val)
            patch["_hits"].append("safe_distance")

        m2 = re.search(r"(转向角|steer|steering).*?([0-9.]+)\s*(deg|degree|degrees|度|rad|弧度)", tl)
        if m2:
            val = float(m2.group(2)); unit = m2.group(3)
            rad = _to_rad(val, unit)
            patch["constraints.delta_max"] = ("set", float(rad))
            patch["constraints.delta_min"] = ("set", -float(rad))
            patch["_hits"].append("steer_limit")

        if re.search(r"(更远|longer|更长视野)", tl):
            patch["horizon"] = ("add_int", 10)
            patch["_hits"].append("longer_horizon")
        if re.search(r"(更短|shorter)", tl):
            patch["horizon"] = ("add_int", -10)
            patch["_hits"].append("shorter_horizon")

        if re.search(r"(更安全|保守|safer|conservative)", tl):
            patch["obstacle.radius"] = ("add", 0.2)
            patch["constraints.delta_max"] = ("mul_clip", 0.8, 0.15, 0.8)
            patch["constraints.delta_min"] = ("sym_from", "constraints.delta_max")
            patch["_hits"].append("conservative")

        if re.search(r"(更灵活|敏捷|agile|nimble)", tl):
            patch["constraints.delta_max"] = ("mul_clip", 1.2, 0.15, 0.8)
            patch["constraints.delta_min"] = ("sym_from", "constraints.delta_max")
            patch["weights.control"] = ("mul_clip_vec2", 0.8, 0.005, 10.0)
            patch["_hits"].append("agile")

        risk = (context or {}).get("risk_hint")
        if risk in ["low", "med", "high"]:
            patch["__risk_level__"] = ("set", risk)
            patch["_hits"].append(f"risk:{risk}")

        return patch

    def parse(self, text, context=None):
        context = context or {}
        if self.provider is None:
            return self._fallback_rules(text, context)

        # 走 LLM，任何异常都回退规则，保证稳定
        prompt = f"""
You are a control-semantic compiler. Given user instruction and context, produce a JSON patch for an MPC DSL.
Keys may include:
  "terminal_scale": ["scale", r]
  "weights.control": ["mul", r]
  "horizon": ["add_int", k]
  "obstacle.radius": ["add", x]
  "constraints.delta_max": ["set", rad]
  "constraints.delta_min": ["set", -rad]
  "__risk_level__": ["set", "low|med|high"]
Return JSON only.
INSTRUCTION: {text}
CONTEXT: {context}
""".strip()

        try:
            raw = self.provider(prompt)
        except Exception as e:
            print(f"[LLMInterpreter] provider error: {e}. Fallback to rules.")
            return self._fallback_rules(text, context)

        # 尝试解析 JSON，不行就回退
        try:
            return json.loads(raw)
        except Exception:
            print("[LLMInterpreter] provider returned non-JSON, fallback to rules.")
            return self._fallback_rules(text, context)
