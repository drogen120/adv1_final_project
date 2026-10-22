#!/usr/bin/env bash
# Run a batch of closed-loop evaluations one after the other, then compare them.
#
#   pixi run bash scripts/run_evals.sh                 # every evaluation in JOBS below
#   pixi run bash scripts/run_evals.sh act_img_291_60k_boundary act_291_envnorm_retry_60k_boundary
#   pixi run bash scripts/run_evals.sh --list          # show the jobs and which are done
#
# Each job starts policy_server, sim_node (sync_mode, as fast as possible, headless) and
# policy_eval, writes data/eval/<name>/ like a manual run (logs in data/eval/<name>/logs/),
# then renders the failures with render_eval_cases.py. The cameras a policy needs are read
# from its config.json and passed to sim_node. A job whose summary.json already exists is
# skipped, so after an interruption just run the script again; a job whose checkpoint is
# missing is skipped with a warning. At the end it prints a table of every job and runs
# grasp_metrics.py on the pairs in COMPARE (paired episode by episode, same seed).
#
# For a quick test of the script itself: EVAL_ROOT=/tmp/x EVAL_EPISODES=2 (output root and
# episodes per job; defaults data/eval and the numbers in JOBS).
set -eu
set -m   # every background job in its own process group, so it can be stopped with kill -INT -pgid

cd "$(dirname "$0")/.."
PORT=5557
ROOT=${EVAL_ROOT:-data/eval}

IMG=outputs/act_img_291/checkpoints
STATE=outputs/act_291_envnorm_retry/checkpoints
BOUNDARY='-p box_rel_yaw_range:=[35.0,55.0]'

# name | checkpoint | seed | episodes | extra sim_node arguments
JOBS=(
  "act_img_291_60k_boundary|$IMG/060000/pretrained_model|100|50|$BOUNDARY"
  "act_291_envnorm_retry_60k_boundary|$STATE/060000/pretrained_model|100|50|$BOUNDARY"
  "act_img_291_60k_seed101|$IMG/060000/pretrained_model|101|100|"
  "act_img_291_40k|$IMG/040000/pretrained_model|100|50|"
)

# Pairs compared by grasp_metrics.py: reference first. act_291_envnorm_retry_60k_100 is the
# earlier seed-101, 100-episode evaluation of the state policy.
COMPARE=(
  "act_291_envnorm_retry_60k_boundary act_img_291_60k_boundary"
  "act_291_envnorm_retry_60k_100 act_img_291_60k_seed101"
  "act_img_291_60k act_img_291_40k"
)

job_field() { echo "$1" | cut -d'|' -f"$2"; }

if [ "${1:-}" = --list ]; then
  for job in "${JOBS[@]}"; do
    name=$(job_field "$job" 1); ckpt=$(job_field "$job" 2)
    state=todo
    [ -f "$ROOT/$name/summary.json" ] && state=done
    [ -d "$ckpt" ] || state="no checkpoint ($ckpt)"
    printf '%-40s seed %-4s %4s episodes  %s\n' "$name" "$(job_field "$job" 3)" "$(job_field "$job" 4)" "$state"
  done
  exit 0
fi

SELECTED=("$@")
selected() {
  [ ${#SELECTED[@]} -eq 0 ] && return 0
  for s in "${SELECTED[@]}"; do [ "$s" = "$1" ] && return 0; done
  return 1
}

set +u   # the ROS setup scripts use unset variables
source ros2_ws/install/setup.bash
set -u

PIDS=()
stop_all() {
  for pid in ${PIDS[@]+"${PIDS[@]}"}; do kill -INT -"$pid" 2>/dev/null || true; done
  for _ in $(seq 50); do   # up to 5 s for a clean shutdown
    alive=0
    for pid in ${PIDS[@]+"${PIDS[@]}"}; do kill -0 "$pid" 2>/dev/null && alive=1; done
    [ $alive = 0 ] && break
    sleep 0.1
  done
  for pid in ${PIDS[@]+"${PIDS[@]}"}; do kill -KILL -"$pid" 2>/dev/null || true; done
  PIDS=()
}
trap stop_all EXIT
trap 'exit 130' INT TERM   # exit runs stop_all through the EXIT trap

run_job() {
  local name=$1 ckpt=$2 seed=$3 episodes=$4 extra=$5
  local out=$ROOT/$name logs=$ROOT/$name/logs
  mkdir -p "$logs"

  local cameras
  cameras=$(python -c 'import json, sys
features = json.load(open(sys.argv[1] + "/config.json"))["input_features"]
print(",".join(k.split(".", 2)[2] for k in features if k.startswith("observation.images.")))' "$ckpt")
  local sim_args=(-p sync_mode:=true -p realtime_factor:=0.0 -p seed:="$seed")
  [ -n "$cameras" ] && sim_args+=(-p cameras:="$cameras")
  # shellcheck disable=SC2206  # extra holds whitespace-separated arguments without spaces
  [ -n "$extra" ] && sim_args+=($extra)

  echo "=== $name: $ckpt, seed $seed, $episodes episodes${cameras:+, cameras $cameras}${extra:+, $extra}"
  echo "    started $(date '+%H:%M:%S'), logs in $logs/"

  .venv-lerobot/bin/python scripts/policy_server.py "$ckpt" --port "$PORT" > "$logs/policy_server.log" 2>&1 &
  local server=$!
  PIDS+=("$server")
  for _ in $(seq 600); do   # loading torch and the policy takes a while the first time
    grep -q '^listening' "$logs/policy_server.log" && break
    kill -0 "$server" 2>/dev/null || { cat "$logs/policy_server.log" >&2; return 1; }
    sleep 0.1
  done
  grep -q '^listening' "$logs/policy_server.log" || { echo "policy server did not start" >&2; return 1; }

  ros2 run so101_mujoco sim_node --ros-args "${sim_args[@]}" > "$logs/sim_node.log" 2>&1 &
  PIDS+=("$!")

  # Foreground: its exit status says whether the evaluation finished.
  if ! ros2 run so101_mujoco policy_eval --ros-args -p num_episodes:="$episodes" \
       -p server_port:="$PORT" -p output_dir:="$out" > "$logs/policy_eval.log" 2>&1; then
    tail -5 "$logs/policy_eval.log" >&2
    stop_all
    return 1
  fi
  stop_all
  grep 'Success rate' "$logs/policy_eval.log" | sed 's/^.*Success rate/    Success rate/'

  python scripts/render_eval_cases.py "$out" --num-success 1 > "$logs/render.log" 2>&1 \
    || echo "    (rendering the videos failed, see $logs/render.log)"
}

FAILED=()
for job in "${JOBS[@]}"; do
  name=$(job_field "$job" 1)
  selected "$name" || continue
  ckpt=$(job_field "$job" 2)
  if [ -f "$ROOT/$name/summary.json" ]; then
    echo "=== $name: already done, skipped (delete $ROOT/$name to rerun)"
    continue
  fi
  if [ ! -d "$ckpt" ]; then
    echo "=== $name: WARNING checkpoint not found ($ckpt), skipped"
    continue
  fi
  run_job "$name" "$ckpt" "$(job_field "$job" 3)" "${EVAL_EPISODES:-$(job_field "$job" 4)}" \
    "$(job_field "$job" 5)" \
    || { echo "=== $name: FAILED"; FAILED+=("$name"); }
done

echo
echo "=== Summary"
printf '%-40s %9s %9s %7s %10s %10s\n' evaluation success by600 lifted regrasped recovered
for name in act_291_envnorm_retry_60k act_291_envnorm_retry_60k_100 act_img_291_60k \
            $(for job in "${JOBS[@]}"; do job_field "$job" 1; done); do
  [ -f "$ROOT/$name/summary.json" ] || continue
  python -c 'import json, sys
s = json.load(open(sys.argv[1] + "/summary.json"))
print("{:40s} {:4d}/{:<4d} {:9d} {:7d} {:10d} {:10d}".format(
    sys.argv[2], s["successes"], s["episodes"], s["successes_by_frame_600"], s["lifted"],
    s["regrasped"], s["recovered"]))' "$ROOT/$name" "$name"
done

for pair in "${COMPARE[@]}"; do
  read -r ref other <<< "$pair"
  [ -f "$ROOT/$ref/rollouts.npz" ] && [ -f "$ROOT/$other/rollouts.npz" ] || continue
  echo
  echo "=== grasp_metrics: $ref vs $other"
  python scripts/grasp_metrics.py "$ROOT/$ref" "$ROOT/$other"
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo
  echo "Jobs that did not finish: ${FAILED[*]}"
  exit 1
fi
