# ====== 基本路径 ======
$task = (Resolve-Path .\dsl\base.json).Path
New-Item -ItemType Directory -Force -Path exp | Out-Null

function Run-Manual($name, $jsonPatch) {
  $patchPath = "patch\$name.json"
  New-Item -ItemType Directory -Force -Path (Split-Path $patchPath) | Out-Null
  $jsonPatch | Set-Content -Encoding utf8NoBOM $patchPath
  $outPrefix = "exp\$name\$name"
  python -m sim.sim_runner $task $patchPath --out $outPrefix --llm none
  # 归档补丁与 tmp
  Copy-Item -Force .\last_patch.json "exp\$name\last_patch_manual.json" -ErrorAction SilentlyContinue
  Copy-Item -Force .\_tmp_task.json  "exp\$name\_tmp_task_manual.json" -ErrorAction SilentlyContinue
}

function Run-LLM($name, $instrText) {
  $outPrefix = "exp\$name\${name}_llm"
  $instr = @"
$instrText
"@
  python -m sim.sim_runner $task $instr --llm ollama --model qwen2.5:7b --k 1 --temp 0 --out $outPrefix --save-llm
  Copy-Item -Force .\last_patch.json "exp\$name\last_patch_llm.json" -ErrorAction SilentlyContinue
  Copy-Item -Force .\_tmp_task.json  "exp\$name\_tmp_task_llm.json" -ErrorAction SilentlyContinue
}

# ====== E1~E6 ======
Run-Manual "E1" '{"obstacle":{"radius":0.6}}'
Run-LLM    "E1"  '{"obstacle":{"radius":0.6}}'

Run-Manual "E2" '{"u_rate_weight":1.0}'
Run-LLM    "E2"  '更强调平滑控制，把 u_rate_weight 提高到 1.0'

Run-Manual "E3" '{"weights":{"terminal_velocity":60}}'
Run-LLM    "E3"  '更靠近目标点末端停车，把 terminal_velocity 罚权提到 60'

Run-Manual "E4" '{"horizon":120}'
Run-LLM    "E4"  '把预测步长缩短到 120'

Run-Manual "E5" '{"insert_midpoint":true}'
Run-LLM    "E5"  '启用中点引导，让轨迹更圆滑'

Run-Manual "E6" '{"speed_cap":{"enabled":true,"d0":0.9,"v_far":1.0,"v_near":0.18}}'
Run-LLM    "E6"  '把靠近目标的限速加强：v_near=0.18，v_far=1.0'
