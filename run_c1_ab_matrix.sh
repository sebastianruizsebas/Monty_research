#!/usr/bin/env bash
#
# C1 A/B matrix: {stock, telemetry, floored} x {1 LM, 5 LMs} on 20 YCB objects.
#
# Runs each cell one at a time; each cell internally parallelises across episodes
# via run_parallel.py. Completed cells are skipped, so the script is resumable
# after a Ctrl-C or a reboot.
#
#   ./run_c1_ab_matrix.sh              # full matrix
#   SMOKE=1 ./run_c1_ab_matrix.sh      # 4 episodes per cell, to prove the wiring
#   NP_5LM=2 ./run_c1_ab_matrix.sh     # hand-tune pool sizes
#
set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------- tunables ---
EPOCHS="${EPOCHS:-3}"        # rotations per object => 20 x EPOCHS episodes/cell
NP_1LM="${NP_1LM:-8}"        # workers for 1-LM cells
NP_5LM="${NP_5LM:-4}"        # workers for 5-LM cells (bigger model, fewer fit)
SMOKE="${SMOKE:-0}"
CONDA_ENV="${CONDA_ENV:-tbp.monty}"

# One BLAS thread per worker. Without this, 8 workers x an OpenMP pool sized to
# all 32 cores thrash each other and the run gets slower as you add workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export MONTY_LOGS="${MONTY_LOGS:-$HOME/tbp/results/monty}"
export MONTY_MODELS="${MONTY_MODELS:-$HOME/tbp/results/monty/pretrained_models}"
export MONTY_DATA="${MONTY_DATA:-$HOME/tbp/data}"

LOGDIR="${MONTY_LOGS}/projects/evidence_eval_runs/logs"
TELDIR="${MONTY_LOGS}/projects/evidence_eval_runs/c1_telemetry"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUNLOG="./c1_ab_matrix_${STAMP}.log"

# ------------------------------------------------------------ conda set-up ---
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

# ------------------------------------------------------------------ matrix ---
# arm | lm_count | learning_module config | num_parallel
MATRIX=(
  "stock|1|evidence_1lm_nn10_dod003|${NP_1LM}"
  "telemetry|1|evidence_1lm_c1_telemetry|${NP_1LM}"
  "floored|1|evidence_1lm_c1_gaussian|${NP_1LM}"
  "stock|5|evidence_5lm_nn10_dod003|${NP_5LM}"
  "telemetry|5|evidence_5lm_c1_telemetry|${NP_5LM}"
  "floored|5|evidence_5lm_c1_gaussian|${NP_5LM}"
)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUNLOG"; }

log "matrix start | epochs=${EPOCHS} np_1lm=${NP_1LM} np_5lm=${NP_5LM} smoke=${SMOKE}"
log "results  -> ${LOGDIR}"
log "telemetry-> ${TELDIR}"

for entry in "${MATRIX[@]}"; do
  IFS='|' read -r arm lms lm_cfg nparallel <<< "$entry"

  if [[ "$lms" == "1" ]]; then
    base_exp="randrot_noise_77obj_dist_agent"
  else
    base_exp="randrot_noise_77obj_5lms_dist_agent"
  fi

  run_name="c1_${arm}_${lms}lm_20obj"
  [[ "$SMOKE" == "1" ]] && run_name="smoke_${run_name}"

  if [[ -f "${LOGDIR}/${run_name}/eval_stats.csv" ]]; then
    log "skip  ${run_name} (eval_stats.csv already present)"
    continue
  fi

  # Only the telemetry arm writes a side-channel. Without this variable the
  # records are held on the class and die with each spawned worker.
  if [[ "$arm" == "telemetry" ]]; then
    export MONTY_C1_TELEMETRY_DIR="${TELDIR}/${run_name}"
    mkdir -p "$MONTY_C1_TELEMETRY_DIR"
  else
    unset MONTY_C1_TELEMETRY_DIR || true
  fi

  extra=()
  [[ "$SMOKE" == "1" ]] && extra+=("episodes=0:4")

  log "start ${run_name} | ${base_exp} | ${lm_cfg} | workers=${nparallel}"
  cell_start=$SECONDS

  python run_parallel.py \
    experiment="${base_exp}" \
    monty/learning_module="${lm_cfg}" \
    env_interface=eval_20obj_random \
    logging=basic_info_evidence_eval_runs \
    num_parallel="${nparallel}" \
    ++experiment.config.n_eval_epochs="${EPOCHS}" \
    ++experiment.config.logging.run_name="${run_name}" \
    "${extra[@]}" >>"$RUNLOG" 2>&1

  log "done  ${run_name} in $((SECONDS - cell_start))s"
done

log "matrix complete"
log "next: ./compare_c1_arms.py"
