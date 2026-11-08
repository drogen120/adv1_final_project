"""A trained LeRobot policy with plain numpy inputs, for policy_server.py and sim_eval.py.

Run from the LeRobot venv (.venv-lerobot). Importing this module drops the pixi environment's
site-packages from sys.path: from a `pixi shell`, PYTHONPATH points at the ROS environment,
whose numpy and libomp clash with the venv's torch (OMP Error #15). Only site-packages go:
the venv may be built on pixi's Python, whose stdlib is under .pixi.
"""

import sys

sys.path = [p for p in sys.path if not ('.pixi' in p and 'site-packages' in p)]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies import get_policy_class, make_pre_post_processors  # noqa: E402


def default_device():
    if torch.cuda.is_available():
        return 'cuda'
    # Small batch-1 models are usually no faster on MPS than on the CPU.
    return 'cpu'


class Policy:
    """Images are {camera: (H, W, 3) uint8 RGB}, as rendered by so101_mujoco.cameras; they
    are scaled to [0, 1] and made channel-first, like LeRobot's video decoding does for the
    training frames. Only the inputs the policy was trained on (its input_features) are
    used, so e.g. a policy trained on images alone never sees environment_state."""

    def __init__(self, path, device=None, n_action_steps=None, temporal_ensemble=None):
        device = device or default_device()
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
        features = config.input_features
        self.uses_env = 'observation.environment_state' in features
        self.cameras = [k.removeprefix('observation.images.') for k in features
                        if k.startswith('observation.images.')]
        self.image_shapes = {c: tuple(features[f'observation.images.{c}'].shape)
                             for c in self.cameras}  # (C, H, W)

    def info(self):
        return {'state': True, 'environment_state': self.uses_env, 'cameras': self.cameras}

    def reset(self):
        """Call at the start of every episode: clears the action queue (or the ensembler)."""
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def _batch(self, state, environment_state=None, images=None):
        batch = {'observation.state': torch.tensor(np.asarray([state]), dtype=torch.float32)}
        if self.uses_env:
            batch['observation.environment_state'] = torch.tensor(
                np.asarray([environment_state]), dtype=torch.float32)
        for cam in self.cameras:
            pixels = np.asarray(images[cam], dtype=np.uint8)
            h, w, c = pixels.shape
            if (c, h, w) != self.image_shapes[cam]:
                raise ValueError(f'{cam}: got a {h}x{w}x{c} image, the policy was trained on '
                                 f'{self.image_shapes[cam]} (C, H, W)')
            pixels = torch.from_numpy(pixels.copy())
            batch[f'observation.images.{cam}'] = (pixels.permute(2, 0, 1).float() / 255.0)[None]
        return self.preprocessor(batch)

    @torch.inference_mode()
    def act(self, state, environment_state=None, images=None):
        """The next action (list of floats), executing chunks as configured
        (n_action_steps / temporal ensembling)."""
        action = self.policy.select_action(self._batch(state, environment_state, images))
        return self.postprocessor(action)[0].cpu().tolist()

    @torch.inference_mode()
    def predict_chunk(self, state, environment_state=None, images=None):
        """The whole predicted chunk, (chunk_size, action_dim) float32 numpy, unnormalized;
        stateless (no queue). Taking its first n_action_steps rows every n_action_steps
        frames is what act() does without temporal ensembling."""
        chunk = self.policy.predict_action_chunk(self._batch(state, environment_state, images))
        return self.postprocessor(chunk)[0].cpu().numpy()

    @torch.inference_mode()
    def predict_chunk_and_embedding(self, state, environment_state=None, images=None):
        """predict_chunk plus the ACT transformer encoder's output for this observation,
        (n_tokens, dim_model) float32 numpy: [latent, state, (env), camera feature tokens...].
        ACT only."""
        captured = {}
        hook = self.policy.model.encoder.register_forward_hook(
            lambda module, inputs, output: captured.setdefault('out', output))
        try:
            chunk = self.predict_chunk(state, environment_state, images)
        finally:
            hook.remove()
        return chunk, captured['out'][:, 0].float().cpu().numpy()
