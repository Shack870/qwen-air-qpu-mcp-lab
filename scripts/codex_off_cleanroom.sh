#!/bin/zsh
set -u

LAB="${QPU_MCP_LAB_ROOT:-$HOME/qwen-air-tests/qpu-mcp-lab}"
LLAMA="${QPU_MCP_LAB_LLAMA_BIN:-$HOME/src/ik_llama.cpp/build-air-iqk-lean/bin/llama-cli}"
MODEL="${QPU_MCP_LAB_MODEL_PATH:-$HOME/qwen-air-tests/models/byteshape-qwen3-30b-a3b-2507/Qwen3-30B-A3B-Instruct-2507-Q3_K_S-2.66bpw.gguf}"
OUTDIR="$LAB/logs/cleanroom-$(date +%Y%m%d-%H%M%S)"
PROMPT='<|im_start|>user
Continue this comma-separated list of Mars facts: red planet, thin atmosphere,<|im_end|>
<|im_start|>assistant
'

mkdir -p "$OUTDIR"

echo "Clean-room Qwen Air benchmark"
echo "Output directory: $OUTDIR"
echo "Started: $(date)"
echo
echo "Close every app you can before this run. Plug in power. Let the Mac sit idle for a minute if it feels warm."
echo "Press RETURN to start, or Ctrl-C to cancel."
read _

run_case() {
  local label="$1"
  local ctx="$2"
  local batch="$3"
  local ubatch="$4"
  local predict="$5"
  shift 5
  local extra_args=("$@")
  local log="$OUTDIR/${label}.log"

  echo
  echo "=== $label ==="
  echo "Log: $log"
  date "+start %Y-%m-%d %H:%M:%S" | tee -a "$log"
  /usr/bin/time -l env \
    VECLIB_MAXIMUM_THREADS=1 \
    OMP_WAIT_POLICY=ACTIVE \
    OMP_DYNAMIC=FALSE \
    caffeinate -dimsu \
    "$LLAMA" \
      -m "$MODEL" \
      -c "$ctx" \
      -b "$batch" \
      -ub "$ubatch" \
      -t 4 \
      -tb 4 \
      --cache-type-k q6_0 \
      --cache-type-v q6_0 \
      -n "$predict" \
      --temp 0.0 \
      -fa 1 \
      -ser 3,1 \
      --ignore-eos \
      --no-display-prompt \
      "${extra_args[@]}" \
      -p "$PROMPT" >> "$log" 2>&1
  local exit_code=$?
  date "+end   %Y-%m-%d %H:%M:%S" | tee -a "$log"
  echo "exit_code=$exit_code" | tee -a "$log"
}

run_case "raw_b2304_ub104_n128" 16384 2304 104 128
sleep 20
run_case "raw_b2368_ub96_n128" 16384 2368 96 128
sleep 20
run_case "np2_b2560_ub96_n96" 16384 2560 96 96 -np 2 -ns 2 -pps
sleep 20
run_case "np2_b2560_ub96_n128" 16384 2560 96 128 -np 2 -ns 2 -pps

echo
echo "=== SCOREBOARD ==="
for log in "$OUTDIR"/*.log; do
  label="${log:t:r}"
  gen="$(grep -E 'eval time .* tokens per second' "$log" | grep -v 'prompt eval' | tail -1 | sed -E 's/.* ([0-9.]+) tokens per second.*/\1/')"
  pp="$(grep -E 'prompt eval time .* tokens per second' "$log" | tail -1 | sed -E 's/.* ([0-9.]+) tokens per second.*/\1/')"
  rss="$(grep -E 'maximum resident set size' "$log" | tail -1 | awk '{print $1}')"
  faults="$(grep -E 'page faults' "$log" | tail -1 | awk '{print $1}')"
  swaps="$(grep -E 'swaps' "$log" | tail -1 | awk '{print $1}')"
  wall="$(grep -E ' real ' "$log" | tail -1 | awk '{print $1}')"
  printf "%-28s gen=%-8s pp=%-8s real=%-8s rss=%-12s faults=%-8s swaps=%s\n" "$label" "${gen:-NA}" "${pp:-NA}" "${wall:-NA}" "${rss:-NA}" "${faults:-NA}" "${swaps:-NA}"
done | sort -k2 -r

echo
echo "Finished: $(date)"
echo "Paste the SCOREBOARD plus any run that looks suspiciously high or low."
