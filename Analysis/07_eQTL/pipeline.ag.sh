#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs.ag"
mkdir -p "$LOG_DIR"

RUN="python ${SCRIPT_DIR}/run.py"

JSON="${SCRIPT_DIR}/data_paths.ag.json"
JSON_FLAG="--json ${JSON}"

# data_paths.ag.json does not define addition-mode files, so no --add-addition.

nohup $RUN evaluate $JSON_FLAG --model bican                          > "$LOG_DIR/bican.log" 2>&1 &
nohup $RUN evaluate $JSON_FLAG --model bican --filter basal_ganglia   > "$LOG_DIR/bican_basal_ganglia.log" 2>&1 &
nohup $RUN evaluate $JSON_FLAG --model bican --filter cortex          > "$LOG_DIR/bican_cortex.log" 2>&1 &
nohup $RUN evaluate $JSON_FLAG --model bican --filter cortex_rna          > "$LOG_DIR/bican_cortex_rna.log" 2>&1 &
nohup $RUN evaluate $JSON_FLAG --model bican --filter basal_ganglia_rna   > "$LOG_DIR/bican_basal_ganglia_rna.log" 2>&1 &
nohup $RUN evaluate $JSON_FLAG --model bican --filter rna               > "$LOG_DIR/bican_rna.log" 2>&1 &

# nohup $RUN evaluate $JSON_FLAG --model bican_gtex                          > "$LOG_DIR/bican_gtex.log" 2>&1 &
# nohup $RUN evaluate $JSON_FLAG --model bican_gtex --filter gtex   > "$LOG_DIR/bican_gtex_basal_ganglia.log" 2>&1 &

nohup $RUN evaluate $JSON_FLAG --model borzoi                         > "$LOG_DIR/borzoi.log" 2>&1 &
nohup $RUN evaluate $JSON_FLAG --model borzoi --filter brain          > "$LOG_DIR/borzoi_brain.log" 2>&1 &
nohup $RUN evaluate $JSON_FLAG --model borzoi --filter basal_ganglia  > "$LOG_DIR/borzoi_basal_ganglia.log" 2>&1 &
nohup $RUN evaluate $JSON_FLAG --model borzoi --filter cortex         > "$LOG_DIR/borzoi_cortex.log" 2>&1 &
nohup $RUN evaluate $JSON_FLAG --model borzoi --filter gtex_brain     > "$LOG_DIR/borzoi_gtex_brain.log" 2>&1 &

# alphagenome_paper: tissue filtering is intrinsic (per-tissue columns),
# --filter is ignored for this model type.
nohup $RUN evaluate $JSON_FLAG --model alphagenome_paper              > "$LOG_DIR/alphagenome_paper.log" 2>&1 &

echo "Launched 9 jobs (JSON=${JSON}). PIDs:"
jobs -p
echo "Logs in: $LOG_DIR"
