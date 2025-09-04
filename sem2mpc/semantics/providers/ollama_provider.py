# -*- coding: utf-8 -*-
import os, json, re, time, uuid
import requests

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

def _extract_json(text: str):
    """从返回里提取第一段合法 JSON。先找 ```json fenced```，找不到再找第一个 { ... }。"""
    m = JSON_FENCE_RE.search(text)
    if m:
        cand = m.group(1).strip()
        try:
            return json.loads(cand)
        except Exception:
            pass
    # 退化：粗略括号匹配
    start = text.find("{")
    if start >= 0:
        # 贪心找末尾 }
        end = text.rfind("}")
        if end > start:
            cand = text[start:end+1]
            try:
                return json.loads(cand)
            except Exception:
                pass
    return None

def make_ollama_provider(
    model: str = "qwen2.5:3b",
    base_url: str = "http://127.0.0.1:11434",
    temperature: float = 0.0,
    num_predict: int = 256,
    seed: int = 42,
    save_dir: str | None = None,
    timeout: int = 120,
    max_retries: int = 2,
):
    """
    返回一个 provider(prompt:str)->str
    - 尽力强制 JSON：优先使用 Ollama 的 {"format":"json"}；若模型不支持，则用 system prompt + 解析器兜底。
    - 自动保存原始输入/输出到 save_dir（若提供）。
    """
    session = requests.Session()
    endpoint = base_url.rstrip("/") + "/api/generate"
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    SYSTEM_INSTRUCTIONS = (
        "You are a strict compiler that MUST output *ONLY* a single JSON object with no extra text.\n"
        "If the instruction is vague, still output a best-effort JSON patch.\n"
        "Do not include explanations, code fences, or markdown. JSON only.\n"
        'Allowed keys and value forms:\n'
        '  "terminal_scale": ["scale", <float>]\n'
        '  "weights.control": ["mul", <float>]\n'
        '  "horizon": ["add_int", <int>]\n'
        '  "obstacle.radius": ["add", <float>]\n'
        '  "constraints.delta_max": ["set", <float in [0.15,0.8]>]\n'
        '  "constraints.delta_min": ["set", <float>]\n'
        '  "__risk_level__": ["set", "low"|"med"|"high"]\n'
        "Return *one* compact JSON object on a single line."
    )

    FEW_SHOT = [
        # 例1：尽快 + 安全距离
        {
            "instruction": "尽快到达目标，安全距离 0.3 m",
            "context": {"risk_hint":"med"},
            "patch": {
                "terminal_scale": ["scale", 1.5],
                "weights.control": ["mul", 0.7],
                "obstacle.radius": ["add", 0.3],
                "__risk_level__": ["set", "med"]
            }
        },
        # 例2：更保守（转向收紧）+ 远一点视野
        {
            "instruction": "更保守一些，并且更远的预测域",
            "context": {"risk_hint":"high"},
            "patch": {
                "constraints.delta_max": ["set", 0.3],
                "constraints.delta_min": ["set", -0.3],
                "horizon": ["add_int", 10],
                "__risk_level__": ["set", "high"]
            }
        },
    ]

    def _log(name: str, data: str):
        if not save_dir:
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        rid = uuid.uuid4().hex[:8]
        path = os.path.join(save_dir, f"ollama_{stamp}_{rid}.{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)

    def _provider(user_prompt: str) -> str:
        # 构造强提示（system+few-shot）
        prompt = (
            f"[SYSTEM]\n{SYSTEM_INSTRUCTIONS}\n"
            f"[CONTEXT]\n(Provide only JSON. No prose.)\n"
        )
        for ex in FEW_SHOT:
            prompt += (
                "\n[EXAMPLE]\n"
                f"INSTRUCTION: {ex['instruction']}\n"
                f"CONTEXT: {json.dumps(ex['context'], ensure_ascii=False)}\n"
                f"OUTPUT_JSON: {json.dumps(ex['patch'], ensure_ascii=False)}\n"
            )
        prompt += f"\n[INSTRUCTION]\n{user_prompt}\n[OUTPUT_JSON]\n"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "seed": int(seed),
                "num_predict": int(num_predict),
            },
            # 对支持的模型，format=json 会直接返回纯 JSON
            "format": "json"
        }

        last_exc = None
        for _ in range(max_retries + 1):
            try:
                resp = session.post(endpoint, json=payload, timeout=timeout)
                # 有些模型不支持 format=json，会 400；此时去掉 format 再试一次
                if resp.status_code >= 400 and "format" in payload:
                    payload.pop("format", None)
                    resp = session.post(endpoint, json=payload, timeout=timeout)

                resp.raise_for_status()
                data = resp.json()
                raw = data.get("response", "")

                _log("request", json.dumps(payload, ensure_ascii=False, indent=2))
                _log("response", raw)

                # 1) format=json 成功 → raw 已是 JSON 字符串
                try:
                    # Qwen 在 format=json 时通常就给 JSON 对象的字符串
                    _ = json.loads(raw)
                    return raw
                except Exception:
                    pass

                # 2) 解析 fenced / 纯文本中的 JSON
                obj = _extract_json(raw)
                if obj is not None:
                    return json.dumps(obj, ensure_ascii=False)

                # 3) 最后兜底：返回原文（让上层 fallback）
                return raw
            except Exception as e:
                last_exc = e
                time.sleep(0.3)

        # 彻底失败：返回一个可解析的错误 JSON，触发上层回退
        err = {"_error": f"Ollama provider failed: {last_exc}"}
        _log("error", json.dumps(err, ensure_ascii=False, indent=2))
        return json.dumps(err, ensure_ascii=False)

    return _provider
