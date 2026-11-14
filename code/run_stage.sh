#!/bin/bash
# ===========================================================================
#  Autonomous, idempotent 3-stage CrossAttnAlign pipeline (fixed WorldStrat).
#    mtxa_aligned -> mtxa_mis_noxa -> mtxa_mis_xa -> paired eval
#  - Each stage auto-resumes from latest ckpt_epN.pt; heals through the
#    ~10-min background clamp (train.py child dies -> loop resumes).
#  - Completion = COUNT(ckpt_ep*.pt) >= TOTAL (train.py saves one file per
#    epoch, 0-indexed ep0..ep(TOTAL-1) -> exactly TOTAL files when done).
#  - Concurrency guard: if <stage>_train.log modified in last GUARD_SEC,
#    another instance is (or was just) training -> wait, then retry. Prevents
#    OOM / ckpt corruption when relaunched while a previous instance is still
#    in its clamp window, or while the manually-started wksJod is still alive.
#  - Idempotent: relaunching picks up exactly where it left off.
#  Usage (repo root):  bash code/run_stage.sh
# ===========================================================================
set -u
BASE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE"
PY="/c/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12.old.12888/python.exe"
CFG="code/configs/external_worldstrat.py"
TOTAL=30
STAGES="mtxa_aligned mtxa_mis_noxa mtxa_mis_xa"
LOG_ROOT="experiments/logs_worldstrat"
CKPT_ROOT="experiments/ckpts_worldstrat"
GUARD_SEC=300   # treat log modified within this window as "another instance alive"

logfresh() {    # returns 0 if $1 modified within GUARD_SEC
  local f="$1"
  [ -f "$f" ] || return 1
  local mtime now diff
  mtime=$(stat -c %Y "$f" 2>/dev/null || echo 0)
  now=$(date +%s)
  diff=$(( now - mtime ))
  [ "$diff" -lt "$GUARD_SEC" ]
}

echo "[pipeline] start $(date)  stages=[$STAGES]  total=$TOTAL"

for STAGE in $STAGES; do
  CKPT_DIR="$CKPT_ROOT/$STAGE"
  LOG="$LOG_ROOT/${STAGE}_train.log"
  mkdir -p "$CKPT_DIR" "$LOG_ROOT"
  echo "[pipeline] === stage $STAGE ==="
  PREV_COUNT=-1
  while true; do
    # ---- concurrency guard (avoid two trainings on the 8GB GPU) ----
    if logfresh "$LOG"; then
      echo "[pipeline][$STAGE] another instance active (log fresh <${GUARD_SEC}s) -> wait 30s"
      sleep 30; continue
    fi
    COUNT=$(ls "$CKPT_DIR"/ckpt_ep*.pt 2>/dev/null | wc -l | tr -d ' ')
    [ -z "$COUNT" ] && COUNT=0
    if [ "$COUNT" -ge "$TOTAL" ]; then
      echo "[pipeline][$STAGE] complete ($COUNT >= $TOTAL epochs)"; break
    fi
    if [ "$COUNT" -eq "$PREV_COUNT" ]; then
      echo "[pipeline][$STAGE] no progress (stuck at $COUNT) -> abort stage to avoid spin"; break
    fi
    PREV_COUNT="$COUNT"
    if [ "$COUNT" -eq 0 ]; then
      echo "[pipeline][$STAGE] fresh ep0 -> $TOTAL"
      "$PY" -u code/train.py --stage "$STAGE" --cfg "$CFG" --epochs "$TOTAL" >>"$LOG" 2>&1
    else
      LATEST=$(ls "$CKPT_DIR"/ckpt_ep*.pt 2>/dev/null | sed 's/.*_ep//; s/\.pt//' | sort -n | tail -1)
      RESUME="$CKPT_DIR/ckpt_ep${LATEST}.pt"
      echo "[pipeline][$STAGE] resume from ep$LATEST (have $COUNT) -> $TOTAL"
      "$PY" -u code/train.py --stage "$STAGE" --cfg "$CFG" --epochs "$TOTAL" --resume "$RESUME" >>"$LOG" 2>&1
    fi
    # train.py exited (killed by clamp OR finished all epochs) -> loop re-evaluates
  done
done

echo "[pipeline] all stages trained -> running paired eval"
"$PY" -u code/eval_paired_mtxa_ws.py >>"$LOG_ROOT/eval_paired.log" 2>&1
echo "[pipeline] eval exit=$?"
echo "DONE $(date)" > "$LOG_ROOT/pipeline_done.marker"
echo "[pipeline] finished $(date)"
