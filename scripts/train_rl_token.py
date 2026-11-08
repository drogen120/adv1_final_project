"""Train the RL token (scripts/rl_token.py) on ACT encoder embeddings, then check what z_rl keeps.

Inputs are output directories of scripts/collect_embeddings.py; the last --val-frac of the
episodes of each directory are held out. Reports the reconstruction on the held-out slots
(R^2 of the standardized embedding, per token group), then probes: small MLPs trained to
predict privileged labels (box centre in the tool axes, jaw yaw error) from z_rl, from the
proprioceptive state alone, from both, and from the mean-pooled raw embedding, on the
slots just before the grasp (frames --probe-frames), where the fine alignment is decided.

    .venv-lerobot/bin/python scripts/train_rl_token.py data/rlt/emb_normal data/rlt/emb_boundary \\
        --output outputs/rlt/rl_token.pt
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lerobot_policy  # noqa: E402,F401  (first: keeps pixi's numpy out of sys.path)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from rl_token import RLToken  # noqa: E402

PROBE_TARGETS = ['approach_mm', 'lateral_mm', 'jaw_mm', 'yaw_err_deg']


def load(dirs, val_frac):
    embs, meta = [], {k: [] for k in ['frame', 'state', 'targets', 'val', 'source']}
    for source, d in enumerate(dirs):
        emb = np.load(os.path.join(d, 'embeddings.npy'), mmap_mode='r')
        slots = np.load(os.path.join(d, 'slots.npz'))
        n = len(slots['episode'])
        embs.append(np.asarray(emb[:n]))
        episodes = slots['episode']
        n_eps = episodes.max() + 1
        meta['val'].append(episodes >= n_eps - max(1, round(val_frac * n_eps)))
        meta['frame'].append(slots['frame'])
        meta['state'].append(slots['state'])
        meta['targets'].append(np.column_stack([1e3 * slots['box_in_tool'], slots['yaw_err']]))
        meta['source'].append(np.full(n, source))
    return np.concatenate(embs), {k: np.concatenate(v) for k, v in meta.items()}


def batches(n, batch_size, rng):
    while True:
        order = rng.permutation(n)
        for i in range(0, n - batch_size + 1, batch_size):
            yield order[i:i + batch_size]


@torch.no_grad()
def encode_all(model, emb, device, batch_size=256):
    model.eval()
    zs, recon_err = [], []
    for i in range(0, len(emb), batch_size):
        x = torch.from_numpy(emb[i:i + batch_size]).to(device)
        z = model.encode(x)
        target = model.normalize(x)
        recon_err.append(((model.decode(z) - target) ** 2).cpu())
        zs.append(z.cpu())
    return torch.cat(zs).numpy(), torch.cat(recon_err).numpy()   # (N, d), (N, T, D)


def token_groups(n_tokens):
    cams = (n_tokens - 2) // 2
    return {'latent': slice(0, 1), 'state': slice(1, 2), 'cam 1 (overview)': slice(2, 2 + cams),
            'cam 2 (wrist)': slice(2 + cams, n_tokens), 'all': slice(0, n_tokens)}


def probe(x_train, y_train, x_val, y_val, device, steps=3000, seed=0):
    """MLP probe; returns per-target R^2 and RMSE on the validation set."""
    torch.manual_seed(seed)
    mx, sx = x_train.mean(0), x_train.std(0) + 1e-6
    my, sy = y_train.mean(0), y_train.std(0) + 1e-6
    xt = torch.tensor((x_train - mx) / sx, dtype=torch.float32, device=device)
    yt = torch.tensor((y_train - my) / sy, dtype=torch.float32, device=device)
    xv = torch.tensor((x_val - mx) / sx, dtype=torch.float32, device=device)
    net = nn.Sequential(nn.Linear(xt.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, yt.shape[1])).to(device)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    for _ in range(steps):
        loss = nn.functional.mse_loss(net(xt), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(xv).cpu().numpy() * sy + my
    rmse = np.sqrt(((pred - y_val) ** 2).mean(0))
    r2 = 1 - ((pred - y_val) ** 2).sum(0) / ((y_val - y_val.mean(0)) ** 2).sum(0)
    return r2, rmse


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('dirs', nargs='+')
    parser.add_argument('--output', required=True)
    parser.add_argument('--val-frac', type=float, default=0.1)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--probe-frames', type=int, nargs=2, default=(100, 160))
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    emb, meta = load(args.dirs, args.val_frac)
    train, val = np.flatnonzero(~meta['val']), np.flatnonzero(meta['val'])
    n_tokens, dim_in = emb.shape[1:]
    print(f'{len(emb)} slots ({len(train)} train, {len(val)} val), {n_tokens} tokens x {dim_in}',
          flush=True)

    model = RLToken(n_tokens, dim_in, d_model=args.d_model, n_heads=args.n_heads,
                    dim_feedforward=4 * args.d_model).to(args.device)
    sample = emb[rng.choice(train, min(len(train), 2000), replace=False)].astype(np.float32)
    model.set_normalization(sample.reshape(-1, dim_in).mean(0), sample.reshape(-1, dim_in).std(0))
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps,
                                                pct_start=0.05)
    emb_train = emb[train]
    start, log_loss = time.perf_counter(), []
    for step, idx in zip(range(1, args.steps + 1), batches(len(train), args.batch_size, rng)):
        model.train()
        loss, _ = model(torch.from_numpy(emb_train[np.sort(idx)]).to(args.device))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        log_loss.append(loss.item())
        if step % 1000 == 0 or step == args.steps:
            _, err = encode_all(model, emb[val], args.device)
            print(f'step {step}: train {np.mean(log_loss):.4f}, val {err.mean():.4f}, '
                  f'{time.perf_counter() - start:.0f} s', flush=True)
            log_loss = []

    # Reconstruction on held-out episodes, R^2 per token group of the standardized embedding.
    z, err = encode_all(model, emb, args.device)
    target_var = {}
    with torch.no_grad():
        for name, sl in token_groups(n_tokens).items():
            x = model.normalize(torch.from_numpy(emb[val][:, sl]).to(args.device)).cpu().numpy()
            target_var[name] = ((x - x.mean(0)) ** 2).mean()
    recon = {name: float(1 - err[val][:, sl].mean() / target_var[name])
             for name, sl in token_groups(n_tokens).items()}
    print('reconstruction R^2 (held-out):', {k: round(v, 3) for k, v in recon.items()})

    # Probes on the slots just before the grasp.
    lo, hi = args.probe_frames
    near = (meta['frame'] >= lo) & (meta['frame'] <= hi)
    tr, va = np.flatnonzero(near & ~meta['val']), np.flatnonzero(near & meta['val'])
    pooled = emb.astype(np.float32).mean(1)
    features = {'state': meta['state'], 'z_rl': z, 'z_rl + state': np.hstack([z, meta['state']]),
                'mean-pooled ACT embedding': pooled}
    y = meta['targets']
    probes = {}
    print(f'probes on frames [{lo}, {hi}] ({len(tr)} train, {len(va)} val slots): R^2 / RMSE')
    print(f'  {"features":28s}' + ''.join(f'{t:>20s}' for t in PROBE_TARGETS))
    for name, x in features.items():
        r2, rmse = probe(x[tr], y[tr], x[va], y[va], args.device)
        probes[name] = {t: {'r2': float(a), 'rmse': float(b)}
                        for t, a, b in zip(PROBE_TARGETS, r2, rmse)}
        print(f'  {name:28s}' + ''.join(f'{a:>10.2f} / {b:6.2f}  ' for a, b in zip(r2, rmse)))
    print('  label std on val: ' + ', '.join(f'{t} {s:.2f}' for t, s in
                                               zip(PROBE_TARGETS, y[va].std(0))))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    report = {'args': vars(args), 'reconstruction_r2': recon, 'probes': probes}
    model.cpu().save(args.output, report=report)
    with open(os.path.splitext(args.output)[0] + '.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
