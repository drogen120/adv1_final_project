"""Online RL on top of the frozen image ACT (after "RL Token", rlt.pdf): env wrapper, actor,
critic and the TD3 agent. scripts/train_rlt.py runs it.

RLTEnv runs an episode of PickPlaceEnv with the base ACT executing its chunks as trained (a
new chunk every n_action_steps frames, from the current observation), and hands control to
the RL actor in the critical phase: every `slot` frames in [phase_start, phase_end) the actor
sees x = (z_rl of the current observation, joint state, phase time) and the reference ã (the
next `slot` actions of the queued ACT chunk, i.e. what the base policy would do) and returns
the `slot` actions to execute instead. The ACT queue keeps advancing, so an actor that
returns ã reproduces the base policy exactly.

Reward (sparse, at the end of the phase): 1 for a strict success (rollouts.strict_success:
the first grasp starts closing by STRICT_GRASP_FRAME with the box within STRICT_LATERAL of
the tool sideways, and the box ends on the target), else 0. A grasp that fails the strict
part ends the episode at phase_end; otherwise the base policy carries on to judge_frame,
where the box is judged (on 268 evaluation episodes with a strict-ok grasp, the box on the
target at frame 600 always agreed with the end of the 900-frame episode; the latest success
frame was 372).

With reward='shaped' (training only), the 0/1 lateral threshold is replaced by
(1 - |lateral| / lateral_scale)+ times success (grasp still in time): run0 with the sparse
reward and whole-chunk actions did not learn, although a privileged lateral correction of
at most 0.05 rad lifts the boundary evaluation from 30/50 to 40/50 strict successes.

The actor is residual: a = ã + Δ, Δ = scale * tanh(MLP(x, ã - state)), with the last layer
initialized to zero (so it starts as the base policy); Δ is the whole chunk (action='chunk',
as in the paper) or one offset per joint added to every action of the chunk
(action='offset'). Its loss is -Q + beta * |Δ/scale|^2, the paper's -Q + beta |a - ã|^2.
The reference input is dropped (zeroed) for half of the actor's training samples, as in the
paper. The critics are TD3 twins Q(x, ã - state, Δ), with chunk-level one-step TD (gamma
per slot).
"""

import copy
import os
import sys
from collections import deque

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'ros2_ws', 'src', 'so101_mujoco'))
from so101_mujoco.expert import is_success  # noqa: E402
from so101_mujoco.rollouts import (STRICT_GRASP_FRAME, STRICT_LATERAL, episode_result,  # noqa: E402
                                   grasp_frames, grasp_lateral)


class RLTEnv:

    def __init__(self, env, policy, token, kinematics, slot=10, phase=(100, 240),
                 judge_frame=500, max_steps=900, reward='strict', lateral_scale=0.006):
        self.env, self.policy, self.token, self.kinematics = env, policy, token, kinematics
        if reward not in ('strict', 'shaped'):
            raise ValueError(reward)
        self.reward, self.lateral_scale = reward, lateral_scale
        self.slot = slot
        self.phase_start, self.phase_end = phase
        self.judge_frame, self.max_steps = judge_frame, max_steps
        self.n_action_steps = policy.config.n_action_steps
        if self.n_action_steps % slot or self.phase_start % slot:
            raise ValueError('slot must divide n_action_steps and phase_start')
        self.queue = deque()

    @property
    def obs_dim(self):
        return self.token.z_dim + 6 + 1


    def reset(self, box_pose=None):
        self.obs = self.env.reset(box_pose)
        self.box_init = self.obs['box_pos']
        self.queue.clear()
        self.frames, self.t = [], 0
        while self.t < self.phase_start:
            self._base_step()
        return self._observe()

    def _apply(self, action):
        action = [float(a) for a in action]
        self.frames.append((self.obs['state'].tolist(), self.obs['environment_state'].tolist(),
                            action, int(round(self.obs['time'] * 1e9))))
        self.obs = self.env.step(action)
        self.t += 1

    def _predict(self):
        """Encoder output of the current observation; refills the ACT queue if empty."""
        images = self.env.render()
        chunk, emb = self.policy.predict_chunk_and_embedding(self.obs['state'], None, images)
        if not self.queue:
            self.queue.extend(chunk[:self.n_action_steps])
        return emb

    def _base_step(self):
        if not self.queue:
            self._predict()
        self._apply(self.queue.popleft())

    @torch.no_grad()
    def _observe(self):
        emb = self._predict()
        z = self.token.encode(torch.from_numpy(emb)[None])[0].numpy()
        phase = (self.t - self.phase_start) / (self.phase_end - self.phase_start)
        state = self.obs['state'].astype(np.float32)
        return {'x': np.concatenate([z, state, [phase]]).astype(np.float32),
                'state': state,
                'ref': np.array(list(self.queue)[:self.slot], dtype=np.float32)}

    def step(self, actions):
        """Execute `slot` actions (slot, 6); returns (next observation or None, reward, done,
        info)."""
        for action in actions:
            self.queue.popleft()
            self._apply(action)
        if self.t < self.phase_end:
            return self._observe(), 0.0, False, {}
        return None, *self._finish()

    def _finish(self):
        grasps = grasp_frames(self.frames, self.kinematics)
        info = {'grasp_frame': grasps[0] if grasps else None,
                'grasp_lateral': grasp_lateral(self.frames[grasps[0]], self.kinematics)
                if grasps else None}
        in_time = bool(grasps and info['grasp_frame'] <= STRICT_GRASP_FRAME)
        info['grasp_ok'] = in_time and abs(info['grasp_lateral']) <= STRICT_LATERAL
        # shaped: (1 - |lateral| / lateral_scale)+ instead of the 0/1 lateral threshold
        shaped = max(0.0, 1 - abs(info['grasp_lateral']) / self.lateral_scale) if in_time else 0.0
        if not (info['grasp_ok'] or (self.reward == 'shaped' and shaped > 0)):
            info['success'] = None   # not judged
            info['strict'] = False
            return 0.0, True, info
        while self.t < self.judge_frame:
            self._base_step()
        info['success'] = is_success(self.obs['box_pos'], self.env.target_pos)
        info['strict'] = info['grasp_ok'] and info['success']
        reward = float(info['strict']) if self.reward == 'strict' else shaped * info['success']
        return reward, True, info

    def finish_episode(self):
        """After done: run the base policy to max_steps and judge like sim_eval; returns
        rollouts.episode_result (for evaluation)."""
        while self.t < self.max_steps:
            self._base_step()
        _, box_final = self.env.judge()
        return episode_result(self.frames, self.box_init, box_final, self.env.target_pos,
                              self.kinematics, 0.0)


def mlp(n_in, n_out, hidden=256, layers=2):
    mods, d = [], n_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, n_out))
    return nn.Sequential(*mods)


class Actor(nn.Module):
    """Residual Δ (B, delta_dim) from observation x (B, obs_dim) and the reference relative
    to the joint state, ã - state (B, ref_dim); keep_ref (B, 1) zeroes the reference input
    where 0 (reference dropout)."""

    def __init__(self, obs_dim, ref_dim, delta_dim, scale):
        super().__init__()
        self.net = mlp(obs_dim + ref_dim, delta_dim)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, x, ref_rel, keep_ref=None):
        if keep_ref is not None:
            ref_rel = ref_rel * keep_ref
        return self.scale * torch.tanh(self.net(torch.cat([x, ref_rel], -1)))


class Critic(nn.Module):
    """Twin Q(x, ã - state, Δ): the executed chunk is ã + Δ (Δ broadcast over the chunk when
    it is one offset per joint)."""

    def __init__(self, obs_dim, ref_dim, delta_dim):
        super().__init__()
        self.q1 = mlp(obs_dim + ref_dim + delta_dim, 1)
        self.q2 = mlp(obs_dim + ref_dim + delta_dim, 1)

    def forward(self, x, ref_rel, delta):
        xa = torch.cat([x, ref_rel, delta], -1)
        return self.q1(xa)[:, 0], self.q2(xa)[:, 0]


class Replay:

    def __init__(self, obs_dim, ref_dim, delta_dim, capacity=200_000):
        self.data = {'x': np.zeros((capacity, obs_dim), np.float32),
                     'ref': np.zeros((capacity, ref_dim), np.float32),
                     'delta': np.zeros((capacity, delta_dim), np.float32),
                     'r': np.zeros(capacity, np.float32),
                     'x2': np.zeros((capacity, obs_dim), np.float32),
                     'ref2': np.zeros((capacity, ref_dim), np.float32),
                     'done': np.zeros(capacity, np.float32)}
        self.n, self.capacity = 0, capacity

    def add(self, **item):
        i = self.n % self.capacity
        for k, v in item.items():
            self.data[k][i] = v
        self.n += 1

    def sample(self, batch_size, rng, device):
        idx = rng.integers(0, min(self.n, self.capacity), batch_size)
        return {k: torch.from_numpy(v[idx]).to(device) for k, v in self.data.items()}

    def __len__(self):
        return min(self.n, self.capacity)


class TD3:
    """action='chunk': Δ is the whole chunk (slot x 6, as in the paper); action='offset': Δ is
    one offset per joint (6), added to every action of the chunk."""

    def __init__(self, obs_dim, slot, scale, action='chunk', gamma=0.99, beta=1.0, lr=3e-4,
                 tau=0.005, target_noise=0.01, target_noise_clip=0.02, ref_dropout=0.5,
                 device='cpu'):
        self.slot, self.action = slot, action
        self.ref_dim = slot * 6
        self.delta_dim = {'chunk': slot * 6, 'offset': 6}[action]
        self.actor = Actor(obs_dim, self.ref_dim, self.delta_dim, scale).to(device)
        self.critic = Critic(obs_dim, self.ref_dim, self.delta_dim).to(device)
        self.actor_targ = copy.deepcopy(self.actor)
        self.critic_targ = copy.deepcopy(self.critic)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr)
        self.gamma, self.beta, self.tau = gamma, beta, tau
        self.target_noise, self.target_noise_clip = target_noise, target_noise_clip
        self.ref_dropout, self.device = ref_dropout, device

    def chunk(self, ref, delta):
        """The executed actions (slot, 6) for the reference (slot, 6) and Δ (numpy)."""
        return ref + delta.reshape(-1, 6)

    @torch.no_grad()
    def act(self, x, ref_rel):
        """Δ for one observation (numpy)."""
        x = torch.from_numpy(x)[None].to(self.device)
        ref_rel = torch.from_numpy(ref_rel)[None].to(self.device)
        return self.actor(x, ref_rel)[0].cpu().numpy()

    def update_critic(self, b):
        with torch.no_grad():
            delta2 = self.actor_targ(b['x2'], b['ref2'])
            noise = (torch.randn_like(delta2) * self.target_noise).clamp(
                -self.target_noise_clip, self.target_noise_clip)
            q1, q2 = self.critic_targ(b['x2'], b['ref2'], delta2 + noise)
            y = b['r'] + self.gamma * (1 - b['done']) * torch.min(q1, q2)
        q1, q2 = self.critic(b['x'], b['ref'], b['delta'])
        loss = nn.functional.mse_loss(q1, y) + nn.functional.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        return loss.item(), q1.mean().item()

    def update_actor(self, b):
        keep = (torch.rand(len(b['x']), 1, device=self.device) >= self.ref_dropout).float()
        delta = self.actor(b['x'], b['ref'], keep)
        q1, _ = self.critic(b['x'], b['ref'], delta)
        penalty = (delta / self.actor.scale).pow(2).mean()
        loss = -q1.mean() + self.beta * penalty
        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        with torch.no_grad():
            for net, targ in ((self.actor, self.actor_targ), (self.critic, self.critic_targ)):
                for p, pt in zip(net.parameters(), targ.parameters()):
                    pt.lerp_(p, self.tau)
        return loss.item(), penalty.item()

    def state_dict(self):
        return {k: getattr(self, k).state_dict() for k in
                ('actor', 'critic', 'actor_targ', 'critic_targ', 'actor_opt', 'critic_opt')}

    def load_state_dict(self, sd):
        for k, v in sd.items():
            getattr(self, k).load_state_dict(v)
