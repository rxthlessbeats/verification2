#!/usr/bin/env bash
# Run experiment steps on the frozen benchmark splits with a timestamped run.log and timing.
set -uo pipefail
SCRIPT=$(realpath "$0")
cd "$(dirname "$SCRIPT")/.."

usage() {
  cat <<'EOF'
Usage: scripts/experiment.sh NAME [STEP...] [OPTION...]
       scripts/experiment.sh NAME status|stop [--output DIR]

Results, run.log and timing.txt go to outputs/NAME. Tasks come from the frozen
train/test split of each benchmark in data/splits/<benchmark>.json (tau2 uses its
own split; the others are split 50/50 the first time they are used). Steps run in
the order given; the default is doctor pilot train. A failed step stops the run.

Steps:
  doctor   readiness checks for the benchmarks; no paid calls
  pilot    first train task per benchmark; must pass before train and test
  train    training partition; conditions default to QERA
  freeze   lock code, prompts, models and settings into protocol.json
  test     held-out test partition, run once; conditions default to QA QEA QRA QERA
Commands:
  status   whether a run is active, its timing so far and the latest log lines
  stop     stop a --detach run and remove its SWE containers

Options:
  --output DIR          experiment folder (default outputs/NAME)
  --conditions "A B"    verifier views for train and test
  --rerun "A B" | all   solve these benchmarks' stored trajectories again
                        (use a new --output)
  --retry-failed        retry tasks that failed earlier in this folder
  --prompts FILE        prompt file (default src/verification2/core/prompts.yaml)
  --train N|all         train tasks per benchmark, balanced over domains (default all)
  --test N|all          held-out tasks per benchmark, likewise (default all)
  --jobs N              tasks solved and verifier checks sent at once (default 1)
  --benchmarks "A B"    benchmarks of this experiment folder; required on its first
                        run, then remembered
  --detach              keep running after the terminal closes; check with status

Examples:
  scripts/experiment.sh tau2 --benchmarks tau2_bench --train 20 --jobs 4 --detach
  scripts/experiment.sh tau2 train --train all --jobs 8 --detach
  scripts/experiment.sh tau2 freeze test --jobs 8 --detach
  scripts/experiment.sh tau2 status
EOF
}

die() {
  echo "$*" >&2
  exit 2
}

case ${1:-} in
  -h | --help) usage && exit 0 ;;
  "" | -*) usage >&2 && exit 2 ;;
esac
NAME=$1
shift
[[ $NAME =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "NAME must be a plain folder name: $NAME"
ARGS=("$@")
OUTPUT=outputs/$NAME
PROMPTS=src/verification2/core/prompts.yaml
STEPS=()
CONDITIONS= RERUN= RETRY= TRAIN= TEST= JOBS= BENCHMARKS= DETACH=0
while (($#)); do
  case $1 in
    doctor | pilot | train | freeze | test | status | stop) STEPS+=("$1") ;;
    --output | --conditions | --rerun | --prompts | --train | --test | --jobs | --benchmarks)
      [[ $# -ge 2 && -n $2 ]] || die "$1 needs a value"
      case $1 in
        --output) OUTPUT=${2%/} ;;
        --conditions) CONDITIONS=$2 ;;
        --rerun) RERUN=$2 ;;
        --prompts) PROMPTS=$2 ;;
        --train) TRAIN=$2 ;;
        --test) TEST=$2 ;;
        --jobs) JOBS=$2 ;;
        --benchmarks) BENCHMARKS=$2 ;;
      esac
      shift
      ;;
    --retry-failed) RETRY=1 ;;
    --detach) DETACH=1 ;;
    -h | --help) usage && exit 0 ;;
    *) die "Unknown step or option: $1 (see --help)" ;;
  esac
  shift
done
for value in "$TRAIN" "$TEST" "$JOBS"; do
  [[ -z $value || $value == all || $value =~ ^[1-9][0-9]*$ ]] || die "Expected a positive number or all: $value"
done
[[ $JOBS != all ]] || die "--jobs needs a number"
((${#STEPS[@]})) || STEPS=(doctor pilot train)
LOG=$OUTPUT/run.log
LOCK=$OUTPUT/run.lock
HINT="scripts/experiment.sh $NAME"
[[ $OUTPUT == "outputs/$NAME" ]] || HINT+=" --output $OUTPUT"
UV=(uv run --no-cache)
[[ -f raw/websites/.env ]] && UV+=(--env-file raw/websites/.env)
UV+=(--env-file .env)
export PYTHONUNBUFFERED=1

doctor_step() {
  local args=(--network --output "$OUTPUT/readiness.json" --benchmarks $BENCHMARKS)
  if "${UV[@]}" python -m verification2.benchmarks.doctor "${args[@]}" >/dev/null; then
    echo "doctor: ready"
    return 0
  fi
  echo "doctor: not ready ($OUTPUT/readiness.json); failing:" "$(
    python3 -c 'import json, sys; print(*(c["name"] for c in json.load(open(sys.argv[1]))["checks"] if not c["ok"]))' \
      "$OUTPUT/readiness.json" 2>/dev/null
  )"
  return 1
}

freeze_step() {
  "${UV[@]}" python -m verification2.experiments.protocol --benchmarks $BENCHMARKS \
    --prompt-file "$PROMPTS" --output "$OUTPUT/protocol.json"
}

partition_step() {
  local limit= args=(--benchmarks $BENCHMARKS --partition "$1" --output "$OUTPUT/$1" --prompt-file "$PROMPTS"
    --cache-dir "$OUTPUT/api_cache")
  case $1 in
    train) args+=(--conditions ${CONDITIONS:-QERA}) limit=$TRAIN ;;
    test) args+=(--conditions ${CONDITIONS:-QA QEA QRA QERA} --protocol "$OUTPUT/protocol.json") limit=$TEST ;;
  esac
  [[ -n $limit && $limit != all ]] && args+=(--limit "$limit")
  [[ -n $JOBS ]] && args+=(--jobs "$JOBS")
  [[ -n $RETRY ]] && args+=(--retry-failed)
  [[ -n $RERUN ]] && args+=(--rerun $RERUN)
  "${UV[@]}" python -m verification2.experiments.run "${args[@]}"
}

steps() {
  echo "experiment $OUTPUT | benchmarks $BENCHMARKS | steps ${STEPS[*]} | code $(git describe --always --dirty 2>/dev/null || echo unknown)"
  local step
  for step in "${STEPS[@]}"; do
    echo "$step"
    case $step in
      doctor) doctor_step ;;
      freeze) freeze_step ;;
      *) partition_step "$step" ;;
    esac || return
  done
}

stamp() {
  local line
  while IFS= read -r line || [[ -n $line ]]; do printf '[%(%F %T)T] %s\n' -1 "$line"; done
}

show_status() {
  [[ -f $LOG ]] || die "No run log in $OUTPUT"
  if flock -n "$LOCK" true; then
    echo "$OUTPUT: not running"
    python3 scripts/timing.py "$LOG"
  else
    echo "$OUTPUT: running"
    python3 scripts/timing.py --now "$LOG"
  fi
  echo "latest:"
  grep '^\[' "$LOG" | tail -n 5 | cut -c1-240
}

stop_run() {
  local pid
  pid=$(cat "$OUTPUT/run.pid" 2>/dev/null) &&
    [[ $(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null) == *experiment.sh* ]] ||
    die "No background run in $OUTPUT"
  pkill -TERM -s "$pid"
  flock -w 60 "$LOCK" true || die "The run in $OUTPUT did not stop within 60 s"
  rm -f "$OUTPUT/run.pid"
  if command -v docker >/dev/null; then
    docker ps -q --filter name=^minisweagent- 2>/dev/null | xargs -r docker rm -f >/dev/null
  fi
  printf '[%(%F %T)T] done (stopped)\n' -1 >>"$LOG"
  echo "Stopped $OUTPUT"
}

if [[ " ${STEPS[*]} " == *" status "* || " ${STEPS[*]} " == *" stop "* ]]; then
  ((${#STEPS[@]} == 1)) || die "status and stop take no other steps"
  if [[ ${STEPS[0]} == status ]]; then show_status; else stop_run; fi
  exit
fi
[[ -f .env ]] || die "Missing .env: copy .env.example to .env and fill in the API keys"
BENCHMARKS=$(echo $BENCHMARKS)
if [[ -f $OUTPUT/benchmarks ]]; then
  saved=$(<"$OUTPUT/benchmarks")
  [[ -z $BENCHMARKS || $BENCHMARKS == "$saved" ]] ||
    die "$OUTPUT uses --benchmarks \"$saved\"; use another NAME for other benchmarks"
  BENCHMARKS=$saved
fi
[[ -n $BENCHMARKS ]] || die "The first run in $OUTPUT needs --benchmarks, e.g. --benchmarks tau2_bench"

mkdir -p "$OUTPUT"
[[ -f $OUTPUT/benchmarks ]] || echo "$BENCHMARKS" >"$OUTPUT/benchmarks"
if [[ -z ${EXPERIMENT_DETACHED:-} ]]; then
  exec 9>>"$LOCK"
  flock -n 9 || die "A run is already active in $OUTPUT; see: $HINT status"
  if ((DETACH)); then
    EXPERIMENT_DETACHED=1 setsid -f bash "$SCRIPT" "$NAME" "${ARGS[@]}" >"$LOG" 2>&1 </dev/null
    echo "Running in the background; log: $LOG"
    echo "Check: $HINT status"
    echo "Stop:  $HINT stop"
    exit 0
  fi
  steps 2>&1 | stamp | tee "$LOG"
  status=$?
  out() { tee -a "$LOG"; }
else
  echo $$ >"$OUTPUT/run.pid"
  trap 'rm -f "$OUTPUT/run.pid"' EXIT
  steps 2>&1 | stamp
  status=$?
  out() { cat; }
fi
printf '[%(%F %T)T] done (exit %s)\n' -1 "$status" | out
python3 scripts/timing.py "$LOG" | tee "$OUTPUT/timing.txt" | out
exit "$status"
