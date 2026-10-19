#!/usr/bin/env bash
# Interactive policy demo: policy server + sim_node with viewer + the policy_demo control panel.
#
#   pixi run bash scripts/demo.sh [--checkpoint DIR] [--speed X] [--port N] [-- policy_server args]
#
#   --checkpoint  pretrained_model directory (default: the recovery-data ACT, 60k steps)
#   --speed       realtime factor to start with, e.g. 0.5 for half speed (default 1.0);
#                 can also be changed from the panel
#   --port        policy server port (default 5555)
#   after --      passed on to scripts/policy_server.py, e.g. -- --temporal-ensemble 0.01
#
# Closing the panel or the viewer, or Ctrl-C here, stops all three. Logs and the recorded
# runs go to data/demo/.
set -eu
set -m   # every background job in its own process group, so it can be stopped with kill -INT -pgid

cd "$(dirname "$0")/.."
ROOT=$(pwd)
CHECKPOINT=outputs/act_291_envnorm_retry/checkpoints/060000/pretrained_model
SPEED=1.0
PORT=5555
SERVER_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint) CHECKPOINT=$2; shift 2 ;;
    --speed) SPEED=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --) shift; SERVER_ARGS=("$@"); break ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$SPEED" in *.*) ;; *) SPEED="$SPEED.0" ;; esac   # ROS float parameters need a '.'

[ -d "$CHECKPOINT" ] || { echo "checkpoint not found: $CHECKPOINT" >&2; exit 1; }
[ -x .venv-lerobot/bin/python ] || { echo ".venv-lerobot not found, see README" >&2; exit 1; }

set +u   # the ROS setup scripts use unset variables
source ros2_ws/install/setup.bash
set -u

LOG_DIR=data/demo/logs
mkdir -p "$LOG_DIR"
PIDS=()
NAMES=()

cleanup() {
  trap - EXIT INT TERM
  for pid in "${PIDS[@]}"; do
    kill -INT -"$pid" 2>/dev/null || true
  done
  for _ in $(seq 50); do   # up to 5 s for a clean shutdown
    alive=0
    for pid in "${PIDS[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    [ $alive = 0 ] && break
    sleep 0.1
  done
  for pid in "${PIDS[@]}"; do kill -KILL -"$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT TERM   # exit runs cleanup through the EXIT trap

echo "policy server: $CHECKPOINT (log: $LOG_DIR/policy_server.log)"
.venv-lerobot/bin/python scripts/policy_server.py "$CHECKPOINT" --port "$PORT" \
  ${SERVER_ARGS[@]+"${SERVER_ARGS[@]}"} > "$LOG_DIR/policy_server.log" 2>&1 &
SERVER=$!
PIDS+=("$SERVER"); NAMES+=(policy_server)
for _ in $(seq 600); do   # loading torch and the policy takes a while the first time
  grep -q '^listening' "$LOG_DIR/policy_server.log" && break
  kill -0 "$SERVER" 2>/dev/null || { cat "$LOG_DIR/policy_server.log" >&2; exit 1; }
  sleep 0.1
done
grep -q '^listening' "$LOG_DIR/policy_server.log" || { echo "policy server did not start" >&2; exit 1; }

echo "sim_node with viewer, speed ${SPEED}x (log: $LOG_DIR/sim_node.log)"
ros2 run --prefix mjpython so101_mujoco sim_node --ros-args \
  -p use_viewer:=true -p sync_mode:=true -p realtime_factor:="$SPEED" \
  > "$LOG_DIR/sim_node.log" 2>&1 &
PIDS+=("$!"); NAMES+=(sim_node)

echo "control panel (log: $LOG_DIR/policy_demo.log)"
ros2 run so101_mujoco policy_demo --ros-args \
  -p server_port:="$PORT" -p speed:="$SPEED" -p output_dir:="$ROOT/data/demo" \
  > "$LOG_DIR/policy_demo.log" 2>&1 &
PIDS+=("$!"); NAMES+=(policy_demo)

# Run until any of the three exits (macOS bash 3.2 has no wait -n).
while true; do
  for i in "${!PIDS[@]}"; do
    kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "${NAMES[$i]} exited, stopping the demo"; exit 0; }
  done
  sleep 0.5
done
