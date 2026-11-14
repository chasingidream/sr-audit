#!/bin/bash
# ===========================================================================
#  E2 multi-seed driver: 3 seeds x { SEN2VENuS 2x (sisr), WorldStrat 6.63x (mtxa_aligned) }
#  Purpose: fill the "single seed" limitation with seed variance on the two axes
#           that do NOT yet have >=3 seeds (PROBA-V already has 3 seeds from P3-D).
#  Design (modelled on run_stage.sh):
#    - serialized on the single 8GB GPU (never two trainings at once);
#    - idempotent + resumable: heals through the ~10-min background clamp
#      (train.py child dies -> loop resumes from latest ckpt_epN.pt);
#    - completion = COUNT(ckpt_ep*.pt) >= 30;
#    - seed isolation via stage suffix _s<N> (train.py strips it for gating;
#      eval_paired_floor.py likewise after the 2026-09-18 patch).
#  Usage (repo root):  bash code/run_e2_multiseed.sh
# ===========================================================================
export PATH="/c/Users/Administrator/.workbuddy/binaries/PortableGit/versions/1.2.0/usr/bin:$PATH"
set -u
BASE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE"
PY="/c/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12.old.12888/python.exe"
TOTAL=30
SEEDS="42 123 7"
GUARD_SEC=300

logfresh() {    # 0 if $1 modified within GUARD_SEC (another instance alive)
  local f="$1"
  [ -f "$f" ] || return 1
  local mtime now diff
  mtime=$(stat -c %Y "$f" 2>/dev/null || echo 0)
  now=$(date +%s)
  diff=$(( now - mtime ))
  [ "$diff" -lt "$GUARD_SEC" ]
}

run_job() {   # cfg base_stage ckpt_root log_root tag
  local CFG="$1" BSTAGE="$2" CKROOT="$3" LROOT="$4" TAG="$5"
  for S in $SEEDS; do
    local STAGE="${BSTAGE}_s${S}"
    local CKPT_DIR="$CKROOT/$STAGE"
    local LOG="$LROOT/${STAGE}_train.log"
    mkdir -p "$CKPT_DIR" "$LROOT"
    echo "[e2] === $TAG seed=$S stage=$STAGE ==="
    local PREV=-1
    while true; do
      if logfresh "$LOG"; then
        echo "[e2][$STAGE] another instance active (log fresh) -> wait 30s"; sleep 30; continue
      fi
      local COUNT
      COUNT=$(ls "$CKPT_DIR"/ckpt_ep*.pt 2>/dev/null | wc -l | tr -d ' ')
      [ -z "$COUNT" ] && COUNT=0
      if [ "$COUNT" -ge "$TOTAL" ]; then
        echo "[e2][$STAGE] complete ($COUNT >= $TOTAL)"; break
      fi
      if [ "$COUNT" -eq "$PREV" ]; then
        echo "[e2][$STAGE] no progress (stuck at $COUNT) -> abort stage"; break
      fi
      PREV="$COUNT"
      if [ "$COUNT" -eq 0 ]; then
        echo "[e2][$STAGE] fresh ep0 -> $TOTAL"
        "$PY" -u code/train.py --stage "$STAGE" --cfg "$CFG" --epochs "$TOTAL" --seed "$S" >>"$LOG" 2>&1
      else
        local LATEST
        LATEST=$(ls "$CKPT_DIR"/ckpt_ep*.pt 2>/dev/null | sed 's/.*_ep//; s/\.pt//' | sort -n | tail -1)
        echo "[e2][$STAGE] resume from ep$LATEST (have $COUNT) -> $TOTAL"
        "$PY" -u code/train.py --stage "$STAGE" --cfg "$CFG" --epochs "$TOTAL" --seed "$S" \
              --resume "$CKPT_DIR/ckpt_ep${LATEST}.pt" >>"$LOG" 2>&1
      fi
    done
  done
}

echo "[e2] start $(date)"
# cheap axis first (SEN2VENuS ~116 train samples), then the heavier WorldStrat axis
run_job code/configs/external_sen2venus.py  sisr          experiments/ckpts_s2v        experiments/logs_s2v        s2v
run_job code/configs/external_worldstrat.py mtxa_aligned  experiments/ckpts_worldstrat  experiments/logs_worldstrat ws

echo "[e2] all training done -> paired eval $(date)"
EVLOG="experiments/logs_e2_eval.log"
"$PY" -u code/eval_paired_floor.py --cfg code/configs/external_sen2venus.py  --stage sisr_s42  --save experiments/paired_s2v_s42.json  >>"$EVLOG" 2>&1
"$PY" -u code/eval_paired_floor.py --cfg code/configs/external_sen2venus.py  --stage sisr_s123 --save experiments/paired_s2v_s123.json >>"$EVLOG" 2>&1
"$PY" -u code/eval_paired_floor.py --cfg code/configs/external_sen2venus.py  --stage sisr_s7   --save experiments/paired_s2v_s7.json   >>"$EVLOG" 2>&1
"$PY" -u code/eval_paired_floor.py --cfg code/configs/external_worldstrat.py --stage mtxa_aligned_s42  --save experiments/paired_ws_s42.json  >>"$EVLOG" 2>&1
"$PY" -u code/eval_paired_floor.py --cfg code/configs/external_worldstrat.py --stage mtxa_aligned_s123 --save experiments/paired_ws_s123.json >>"$EVLOG" 2>&1
"$PY" -u code/eval_paired_floor.py --cfg code/configs/external_worldstrat.py --stage mtxa_aligned_s7   --save experiments/paired_ws_s7.json   >>"$EVLOG" 2>&1
"$PY" -u code/e2_multiseed_summary.py >>"$EVLOG" 2>&1

# make sure vcomp140.dll is restored even if a child was hard-killed
"$PY" -c "import os; a=r'C:\Windows\System32\vcomp140.dll'; h=a+'.hide'; os.rename(h,a) if (os.path.exists(h) and not os.path.exists(a)) else None; print('[e2] vcomp140 restored/ok')"

echo "DONE $(date)" > experiments/e2_done.marker
echo "[e2] finished $(date)"
