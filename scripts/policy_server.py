"""Serve a trained LeRobot policy over a local TCP socket, for `ros2 run so101_mujoco policy_eval`.

LeRobot (and torch) stay out of the ROS environment: this script runs in its own venv and
the ROS node talks to it with newline-delimited JSON, one request per line:

    {"reset": true}                                          -> {"ok": true}
    {"state": [6 floats], "environment_state": [5 floats]}   -> {"action": [6 floats]}

Send a reset at the start of every episode: it clears the policy's action queue (or its
temporal ensembler). Only the standard library is needed on the client side.

    uv venv -p 3.12 .venv-lerobot
    VIRTUAL_ENV=.venv-lerobot uv pip install 'lerobot[dataset,training]==0.6.1'
    .venv-lerobot/bin/python scripts/policy_server.py \
        outputs/train/act_200/checkpoints/last/pretrained_model

--n-action-steps and --temporal-ensemble override how the trained chunk is executed;
neither changes the weights, so one checkpoint can be evaluated both ways.
"""

import argparse
import functools
import json
import socket
import socketserver
import sys
import time

# From a `pixi shell`, PYTHONPATH points at the ROS environment's site-packages, whose numpy
# and libomp clash with the venv's torch (OMP Error #15). Keep the two environments apart.
# Only drop site-packages: the venv may be built on pixi's Python, whose stdlib is under .pixi.
sys.path = [p for p in sys.path if not ('.pixi' in p and 'site-packages' in p)]

print = functools.partial(print, flush=True)  # the log is usually redirected to a file

import torch  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies import get_policy_class, make_pre_post_processors  # noqa: E402


class Policy:

    def __init__(self, path, device, n_action_steps=None, temporal_ensemble=None):
        config = PreTrainedConfig.from_pretrained(path)
        config.device = device
        if temporal_ensemble is not None:
            config.temporal_ensemble_coeff = temporal_ensemble
            config.n_action_steps = 1
        elif n_action_steps is not None:
            config.n_action_steps = n_action_steps
        self.policy = get_policy_class(config.type).from_pretrained(path, config=config)
        self.policy.to(device)
        self.device = device
        override = {'device_processor': {'device': device}}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config, pretrained_path=path, preprocessor_overrides=override,
            postprocessor_overrides=override)
        self.config = config

    def reset(self):
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    @torch.inference_mode()
    def act(self, state, environment_state):
        batch = {
            'observation.state': torch.tensor([state], dtype=torch.float32),
            'observation.environment_state': torch.tensor([environment_state], dtype=torch.float32),
        }
        action = self.policy.select_action(self.preprocessor(batch))
        return self.postprocessor(action)[0].cpu().tolist()


def make_handler(policy):

    class Handler(socketserver.StreamRequestHandler):

        def setup(self):
            super().setup()
            # Small request/reply messages: without this, Nagle + delayed ACKs add ~40 ms a step.
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def handle(self):
            print(f'client connected: {self.client_address}')
            n, busy = 0, 0.0
            for line in self.rfile:
                request = json.loads(line)
                start = time.perf_counter()
                if request.get('reset'):
                    policy.reset()
                    reply = {'ok': True}
                else:
                    reply = {'action': policy.act(request['state'], request['environment_state'])}
                    n += 1
                    busy += time.perf_counter() - start
                self.wfile.write((json.dumps(reply) + '\n').encode())
            if n:
                print(f'client disconnected after {n} steps, {1e3 * busy / n:.1f} ms/step')

    return Handler


def default_device():
    if torch.cuda.is_available():
        return 'cuda'
    # Small batch-1 models are usually no faster on MPS than on the CPU.
    return 'cpu'


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('checkpoint', help='a pretrained_model directory written by lerobot-train')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--device', default=default_device())
    parser.add_argument('--n-action-steps', type=int,
                        help='actions executed per predicted chunk (default: as trained)')
    parser.add_argument('--temporal-ensemble', type=float, metavar='COEFF',
                        help='use temporal ensembling with this coefficient (ACT uses 0.01)')
    args = parser.parse_args()

    policy = Policy(args.checkpoint, args.device, args.n_action_steps, args.temporal_ensemble)
    c = policy.config
    print(f'{c.type} policy from {args.checkpoint} on {args.device}: chunk_size={c.chunk_size}, '
          f'n_action_steps={c.n_action_steps}, temporal_ensemble_coeff={c.temporal_ensemble_coeff}')
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(('127.0.0.1', args.port), make_handler(policy)) as server:
        print(f'listening on 127.0.0.1:{args.port}')
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
