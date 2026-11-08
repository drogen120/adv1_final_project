"""Judging and saving closed-loop episodes, shared by policy_eval (ROS) and scripts/sim_eval.py.

An episode is a list of frames (state, environment_state, action, stamp_ns): the joint
positions and expert.box_features of a published state, the joint targets the policy
answered it with, and the state's stamp. Only depends on numpy and mujoco.
"""

import numpy as np

from .cameras import JOINT_NAMES
from .expert import is_success

# Strict success (online RL, criterion A): besides ending on the target, the first grasp
# starts closing by STRICT_GRASP_FRAME (the expert always starts at frame 156; the image ACT
# hesitates over the box past frame 200 in ~20% of the |g| >= 30 deg episodes) with the box
# centre within STRICT_LATERAL (m) of the tool along the tool y axis (sideways in the jaw
# plane; the expert is within 0.2 mm, and the offset correlates with the placement error).
STRICT_GRASP_FRAME = 200
STRICT_LATERAL = 0.003


def grasp_frames(frames, kinematics, threshold=0.5, max_dist=0.04, max_height=0.05):
    """The grasps: frames where the gripper command starts closing (last frame >= 0.9 rad
    before it crosses threshold) and the tool is then within max_dist (m) of the box
    horizontally and below max_height (m), i.e. not when the gripper closes on the way back
    to the start pose. kinematics is a kinematics.TopDownIK of the scene."""
    grip = np.array([action[5] for _, _, action, _ in frames])
    starts = []
    for i in np.flatnonzero((grip[:-1] >= threshold) & (grip[1:] < threshold)) + 1:
        state, env, _, _ = frames[i]
        tool, _ = kinematics.forward(np.array(state[:5]))
        if np.linalg.norm(tool[:2] - env[:2]) < max_dist and tool[2] < max_height:
            opened = np.flatnonzero(grip[:i] >= 0.9)
            starts.append(int(opened[-1]) + 1 if len(opened) else i)
    return starts


def grasp_attempts(frames, kinematics, **kwargs):
    """Number of grasps, see grasp_frames."""
    return len(grasp_frames(frames, kinematics, **kwargs))


def grasp_lateral(frame, kinematics):
    """Box centre minus tool position along the tool y axis (m) in a frame (state, env, ...)."""
    state, env = frame[0], frame[1]
    tool, rot = kinematics.forward(np.array(state[:5]))
    return float((np.asarray(env[:3]) - tool) @ rot[:, 1])


def success_frame(frames, target_pos):
    """First frame from which the box is on the target (expert.is_success) to the end, None
    if it does not end there."""
    ok = [is_success(env[:3], target_pos) for _, env, _, _ in frames]
    if not ok[-1]:
        return None
    return len(ok) - int(np.argmin(ok[::-1])) if not all(ok) else 0


def episode_result(frames, box_init, box_final, target_pos, kinematics, policy_time):
    """The line of episodes.jsonl for one episode (box positions in m, policy_time in s)."""
    box_init, box_final = np.asarray(box_init), np.asarray(box_final)
    max_box_z = max(env[2] for _, env, _, _ in frames)
    success = is_success(box_final, target_pos)
    grasps = grasp_frames(frames, kinematics)
    first = grasps[0] if grasps else None
    lateral = grasp_lateral(frames[first], kinematics) if grasps else None
    return {
        'box_init': box_init.round(4).tolist(),
        'box_final': box_final.round(4).tolist(),
        'success': success,
        'target_dist': round(float(np.linalg.norm(box_final[:2] - target_pos[:2])), 4),
        'lifted': bool(max_box_z > 0.04),   # box centre rests at 0.02 m
        'grasp_attempts': len(grasps),
        'grasp_frame': first,
        'grasp_lateral': None if lateral is None else round(lateral, 4),
        'strict_success': bool(success and grasps and first <= STRICT_GRASP_FRAME
                               and abs(lateral) <= STRICT_LATERAL),
        'success_frame': success_frame(frames, target_pos),
        'frames': len(frames),
        'policy_ms': round(1e3 * policy_time / max(1, len(frames)), 2),
    }


def summarize(results):
    """summary.json of a list of episode_result."""
    n = len(results)
    n_success = sum(r['success'] for r in results)
    n_strict = sum(r['strict_success'] for r in results)
    return {'episodes': n, 'success_rate': n_success / n, 'successes': n_success,
            'strict_success_rate': n_strict / n, 'strict_successes': n_strict,
            'successes_by_frame_600': sum(r['success'] and r['success_frame'] < 600
                                          for r in results),
            'lifted': sum(r['lifted'] for r in results),
            'regrasped': sum(r['grasp_attempts'] > 1 for r in results),
            'recovered': sum(r['grasp_attempts'] > 1 and r['success'] for r in results)}


def save_rollouts(path, episode_frames):
    """Write the frames of each episode to path, in the format of export_dataset."""
    columns = {k: [] for k in ['observation_state', 'observation_environment_state', 'action',
                               'episode_index', 'frame_index', 'timestamp']}
    for episode, frames in enumerate(episode_frames):
        state, env, action, stamps = map(np.array, zip(*frames))
        columns['observation_state'].append(state)
        columns['observation_environment_state'].append(env)
        columns['action'].append(action)
        columns['episode_index'].append(np.full(len(frames), episode))
        columns['frame_index'].append(np.arange(len(frames)))
        columns['timestamp'].append((stamps - stamps[0]) * 1e-9)
    arrays = {k: np.concatenate(v) for k, v in columns.items()}
    for k in ['observation_state', 'observation_environment_state', 'action', 'timestamp']:
        arrays[k] = arrays[k].astype(np.float32)
    np.savez_compressed(path, **arrays, fps=50.0, joint_names=np.array(JOINT_NAMES))
