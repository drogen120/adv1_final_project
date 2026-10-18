"""Coverage statistics of the exported demonstrations: how much of the task do they span?

The only thing that changes between episodes is the box's starting pose (position on the
floor and yaw modulo 90 degrees); the start pose of the arm and the place target are fixed.
So coverage is measured in that task space, against the distribution sim_node samples
from (radius, azimuth and yaw uniform), and then in joint space, to see how much of that
variety reaches the trajectories the policy learns from. Reads the .npz written by
`ros2 run so101_mujoco export_dataset`; writes PNGs and prints a summary.

    pixi run python scripts/analyze_demos.py                      # figures in data/analysis/
    pixi run python scripts/analyze_demos.py --radius 0.17 0.25   # demos 0-199 (r from 0.17 m)

Pass the box_radius_range / box_azimuth_range the demos were collected with if they are
not sim_node's defaults.
"""

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle, Wedge  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_XY = (0.22, -0.12)     # 'target' site in scene_pick_place.xml
CUBE = 0.04                   # m, box edge length
# Segments of PickPlaceExpert.plan (s), to label the phases of an episode.
PHASES = [('approach', 2.0), ('descend', 1.0), ('grasp', 0.6), ('lift', 1.0),
          ('carry', 1.6), ('lower', 0.8), ('release', 0.5), ('retreat', 0.6), ('return', 1.5)]

# Colors (reference data-viz palette, light mode).
BLUE = '#2a78d6'
RED = '#e34948'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8984'
GRID = '#e4e3df'
SURFACE = '#fcfcfb'
SEQ = LinearSegmentedColormap.from_list('seq_blue', ['#eef4fc', '#9cc2ee', BLUE, '#163f73'])


def style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': MUTED, 'axes.labelcolor': INK_2, 'axes.titlecolor': INK,
        'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.labelsize': 9, 'xtick.color': INK_2, 'ytick.color': INK_2,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
        'axes.axisbelow': True, 'legend.frameon': False, 'legend.fontsize': 8,
        'text.color': INK, 'font.size': 9,
    })


# --- task space -------------------------------------------------------------------------

def task_params(env0):
    """[x, y, z, cos 4yaw, sin 4yaw] at episode start -> (xy, radius, azimuth deg, yaw deg mod 90)."""
    xy = env0[:, :2]
    r = np.linalg.norm(xy, axis=1)
    az = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
    yaw = np.degrees(np.arctan2(env0[:, 4], env0[:, 3])) / 4  # in (-45, 45]
    return xy, r, az, yaw


def sample_task(rng, n, radius, azimuth):
    """Box poses drawn the way sim_node._randomize_box draws them."""
    r = rng.uniform(*radius, n)
    az = np.radians(rng.uniform(*azimuth, n))
    yaw = rng.uniform(-45, 45, n)
    return np.column_stack([r * np.cos(az), r * np.sin(az)]), yaw


def nearest(q_xy, q_yaw, xy, yaw):
    """Per query: distance (m) to the nearest demo, and the (pos, yaw) gaps to every demo."""
    dpos = np.linalg.norm(q_xy[:, None] - xy[None], axis=2)
    dyaw = np.abs((q_yaw[:, None] - yaw[None] + 45) % 90 - 45)
    return dpos, dyaw


def covered(dpos, dyaw, pos_tol, yaw_tol):
    """Fraction of queries with a demo within pos_tol (m) and yaw_tol (deg) at once."""
    return np.mean(np.any((dpos <= pos_tol) & (dyaw <= yaw_tol), axis=1))


def ks_uniform(x, low, high):
    """Kolmogorov-Smirnov statistic and asymptotic p-value against U(low, high)."""
    x = np.sort((x - low) / (high - low))
    n = len(x)
    i = np.arange(1, n + 1)
    d = max(np.max(i / n - x), np.max(x - (i - 1) / n))
    t = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    k = np.arange(1, 101)
    p = float(np.clip(2 * np.sum((-1) ** (k - 1) * np.exp(-2 * k ** 2 * t ** 2)), 0, 1))
    return d, p


def plot_task_space(path, xy, r, az, yaw, radius, azimuth, rng):
    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.55, 1], hspace=0.65, wspace=0.18)
    ax = fig.add_subplot(gs[:, 0])

    # Background: distance from each point of the sampling region to the nearest demo.
    rr, aa = np.meshgrid(np.linspace(*radius, 160), np.radians(np.linspace(*azimuth, 160)))
    gx, gy = rr * np.cos(aa), rr * np.sin(aa)
    gap = np.linalg.norm(np.stack([gx, gy], -1)[..., None, :] - xy, axis=-1).min(-1)
    mesh = ax.pcolormesh(gx, gy, gap * 100, cmap=SEQ, shading='gouraud', rasterized=True)
    cb = fig.colorbar(mesh, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label('distance to nearest demo (cm)')
    cb.outline.set_visible(False)
    ax.add_patch(Wedge((0, 0), radius[1], *azimuth, width=radius[1] - radius[0], fill=False,
                       ec=INK_2, lw=1.2, ls='--'))

    # Demos: a dot at the box centre and a tick along one face normal (yaw mod 90).
    t = np.radians(yaw)
    half = CUBE / 2
    ax.plot(np.stack([xy[:, 0], xy[:, 0] + half * np.cos(t)]),
            np.stack([xy[:, 1], xy[:, 1] + half * np.sin(t)]), color=INK, lw=1.2)
    ax.scatter(*xy.T, s=26, color=INK, ec=SURFACE, lw=1.5, zorder=3)
    ax.add_patch(Rectangle((TARGET_XY[0] - 0.03, TARGET_XY[1] - 0.03), 0.06, 0.06,
                           color=RED, alpha=0.55, lw=0))
    ax.annotate('place target', TARGET_XY, xytext=(0, -28), textcoords='offset points',
                ha='center', color=INK_2, fontsize=8)
    ax.scatter([0], [0], marker='s', s=60, color=MUTED)
    ax.annotate('arm base', (0, 0), xytext=(6, -12), textcoords='offset points',
                color=INK_2, fontsize=8)
    ax.annotate('sampling region', (radius[1] * np.cos(np.radians(azimuth[1])),
                                    radius[1] * np.sin(np.radians(azimuth[1]))),
                xytext=(-10, 8), textcoords='offset points', ha='right', color=INK_2, fontsize=8)
    ax.set_aspect('equal')
    ax.set_xlim(-0.02, 0.28)
    ax.set_ylim(-0.17, 0.19)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Initial box poses ({len(xy)} demos, tick = yaw mod 90°)')

    # Marginals against what sim_node samples from.
    for row, (vals, rng_, label, unit) in enumerate([
            (r * 100, np.array(radius) * 100, 'Radius', 'cm'),
            (az, azimuth, 'Azimuth', 'deg'),
            (yaw, (-45, 45), 'Yaw mod 90', 'deg')]):
        h = fig.add_subplot(gs[row, 1])
        bins = np.linspace(*rng_, 11)
        h.hist(vals, bins=bins, color=BLUE, ec=SURFACE, lw=2)
        h.axhline(len(vals) / 10, color=INK_2, ls='--', lw=1)
        h.text(rng_[1], len(vals) / 10, ' uniform', va='center', color=INK_2, fontsize=8)
        d, p = ks_uniform(vals, *rng_)
        h.set_title(f'{label}: KS D={d:.2f}, p={p:.2f}', fontsize=10)
        h.set_xlabel(unit)
        h.set_ylabel('demos')
        h.grid(axis='x', visible=False)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_coverage(path, xy, yaw, radius, azimuth, rng, n_query=4000):
    q_xy, q_yaw = sample_task(rng, n_query, radius, azimuth)
    dpos, dyaw = nearest(q_xy, q_yaw, xy, yaw)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={'wspace': 0.32})

    # (a) Position only: CDF of the distance to the nearest demo.
    ax = axes[0]
    nn = np.sort(dpos.min(1)) * 100
    ax.plot(nn, np.linspace(0, 1, len(nn)), color=BLUE, lw=2)
    for q in (0.5, 0.9):
        v = np.quantile(nn, q)
        ax.plot([v, v], [0, q], color=MUTED, lw=1, ls=':')
        ax.annotate(f'{q:.0%} within {v:.1f} cm', (v, q), xytext=(6, -4),
                    textcoords='offset points', color=INK_2, fontsize=8)
    ax.set_xlabel('distance from a random box position to the nearest demo (cm)')
    ax.set_ylabel('fraction of task space')
    ax.set_ylim(0, 1.02)
    ax.set_title('Position coverage')

    # (b) Position and yaw: fraction with a demo close in both.
    ax = axes[1]
    pos_tols = np.array([0.5, 1, 1.5, 2, 3, 4]) / 100
    yaw_tols = np.array([2.5, 5, 10, 15, 20, 45])
    grid = np.array([[covered(dpos, dyaw, p, y) for p in pos_tols] for y in yaw_tols])
    im = ax.imshow(grid * 100, cmap=SEQ, vmin=0, vmax=100, origin='lower', aspect='auto')
    for (i, j), v in np.ndenumerate(grid):
        ax.text(j, i, f'{v:.0%}', ha='center', va='center', fontsize=8,
                color=SURFACE if v > 0.55 else INK)
    ax.set_xticks(range(len(pos_tols)), [f'{p * 100:g}' for p in pos_tols])
    ax.set_yticks(range(len(yaw_tols)), [f'{y:g}' for y in yaw_tols])
    ax.set_xlabel('position tolerance (cm)')
    ax.set_ylabel('yaw tolerance (deg, 45 = any)')
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title('Covered: a demo within both tolerances')

    # (c) How coverage grows with more demos drawn the same way (median of 20 draws).
    ax = axes[2]
    ns = np.array([10, 25, 50, 100, 200, 400])
    tols = [(0.01, 10), (0.02, 10), (0.01, 5)]
    for (pt, yt), shade in zip(tols, ['#163f73', BLUE, '#9cc2ee']):
        med = []
        for n in ns:
            vals = []
            for _ in range(20):
                s_xy, s_yaw = sample_task(rng, n, radius, azimuth)
                vals.append(covered(*nearest(q_xy[:1000], q_yaw[:1000], s_xy, s_yaw), pt, yt))
            med.append(np.median(vals))
        ax.plot(ns, med, color=shade, lw=2, marker='o', ms=4)
        ax.annotate(f'{pt * 100:g} cm, {yt:g}°', (ns[-1], med[-1]), xytext=(6, 0),
                    textcoords='offset points', va='center', color=INK_2, fontsize=8)
        ax.scatter([len(xy)], [covered(dpos, dyaw, pt, yt)], s=70, color=shade,
                   ec=INK, lw=1.2, zorder=3)
    ax.axvline(len(xy), color=MUTED, lw=1, ls=':')
    ax.text(len(xy), 1.0, ' this dataset', color=INK_2, fontsize=8, va='top')
    ax.set_xscale('log')
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.minorticks_off()
    ax.set_xlim(8, 1100)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('number of demos (log scale)')
    ax.set_ylabel('fraction of task space covered')
    ax.set_title('Coverage vs. dataset size')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return dpos, dyaw


# --- joint space ------------------------------------------------------------------------

def phase_bounds(actions, fps):
    """Frame indices where the expert's segments start, aligned to the first commanded move."""
    moving = np.flatnonzero(np.any(np.abs(np.diff(actions[0], axis=0)) > 1e-6, axis=1))
    start = moving[0] if len(moving) else 0
    return start + np.round(np.cumsum([0] + [d for _, d in PHASES]) * fps).astype(int)


def shade_phases(ax, bounds, fps, labels=False):
    for i, (name, _) in enumerate(PHASES):
        a, b = bounds[i] / fps, bounds[i + 1] / fps
        if i % 2:
            ax.axvspan(a, b, color=GRID, alpha=0.45, lw=0)
        if labels:
            ax.text((a + b) / 2, 1.0, name, transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=7, color=INK_2)


def plot_joint_space(path, states, actions, names, fps):
    n_ep, n_t, n_j = states.shape
    t = np.arange(n_t) / fps
    bounds = phase_bounds(actions, fps)
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.1], hspace=0.55, wspace=0.18)
    for j in range(n_j):
        ax = fig.add_subplot(gs[j // 2, j % 2])
        shade_phases(ax, bounds, fps)
        ax.plot(t, states[:, :, j].T, color=BLUE, lw=0.8, alpha=0.35)
        ax.set_title(names[j])
        ax.set_ylabel('rad')
        ax.set_xlim(0, t[-1])
        ax.grid(axis='x', visible=False)
        if j >= 4:
            ax.set_xlabel('time (s)')

    # How different the episodes are at each moment: spread across episodes per joint.
    ax = fig.add_subplot(gs[3, :])
    shade_phases(ax, bounds, fps, labels=True)
    spread = states.std(axis=0)  # (T, J)
    order = np.argsort(-spread.mean(0))
    shades = ['#163f73', BLUE, '#6fa6e6', '#9cc2ee', MUTED, '#c3c2b7']
    for k, j in enumerate(order):
        ax.plot(t, spread[:, j], color=shades[k], lw=1.6)
    i_max = np.argmax(spread.mean(1))
    ax.set_xlim(0, t[-1])
    ax.set_xlabel('time (s)')
    ax.set_ylabel('std across episodes (rad)')
    ax.grid(axis='x', visible=False)
    ax.set_title(f'Variation between episodes (largest at {t[i_max]:.1f} s)', pad=14)
    ax.legend([plt.Line2D([], [], color=shades[k], lw=1.6) for k in range(n_j)],
              [names[j] for j in order], ncol=n_j, loc='upper left', bbox_to_anchor=(0, -0.22))
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return spread, bounds


def plot_episodes(path, states, r, az, yaw):
    n_ep = len(states)
    # The expert closes the jaws across the pair of faces closest to perpendicular to the
    # arm, so what shapes the trajectory is the box yaw relative to the arm's direction.
    grasp = (yaw - az + 45) % 90 - 45
    params = {'radius': r, 'azimuth': az, 'yaw': yaw, 'grasp angle': grasp}
    flat = states.reshape(n_ep, -1)
    # RMS joint difference between two episodes, frame by frame (all episodes are aligned).
    dist = np.sqrt(((states[:, None] - states[None]) ** 2).mean(axis=(2, 3)))
    centred = flat - flat.mean(0)
    _, sv, vt = np.linalg.svd(centred, full_matrices=False)
    explained = sv ** 2 / np.sum(sv ** 2)
    pcs = centred @ vt[:2].T

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={'wspace': 0.35})
    ax = axes[0]
    order = np.argsort(grasp)
    im = ax.imshow(dist[np.ix_(order, order)], cmap=SEQ)
    ax.set_xlabel('episode (sorted by grasp angle)')
    ax.set_ylabel('episode (sorted by grasp angle)')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label('RMS joint difference (rad)')
    cb.outline.set_visible(False)
    ax.set_title('Trajectory distance')

    ax = axes[1]
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=grasp, cmap=SEQ, s=40, ec=INK_2, lw=0.6)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label('grasp angle: box yaw - azimuth, mod 90 (deg)')
    cb.outline.set_visible(False)
    drivers = []
    for k in range(2):
        corr = {n: np.corrcoef(v, pcs[:, k])[0, 1] for n, v in params.items()}
        name = max(corr, key=lambda n: abs(corr[n]))
        drivers.append((name, corr[name]))
    ax.set_xlabel(f'PC1 ({explained[0]:.0%} of variance, r={drivers[0][1]:+.2f} with {drivers[0][0]})')
    ax.set_ylabel(f'PC2 ({explained[1]:.0%}, r={drivers[1][1]:+.2f} with {drivers[1][0]})')
    ax.set_title('Episodes in trajectory space (PCA)')

    # Do nearby start poses give nearby trajectories? (a smooth map is easier to learn)
    ax = axes[2]
    iu = np.triu_indices(n_ep, 1)
    xy = np.column_stack([r * np.cos(np.radians(az)), r * np.sin(np.radians(az))])
    dpos = np.linalg.norm(xy[:, None] - xy[None], axis=2)[iu] * 100
    dyaw = np.abs((yaw[:, None] - yaw[None] + 45) % 90 - 45)[iu]
    sc = ax.scatter(dpos, dist[iu], c=dyaw, cmap=SEQ, s=10, lw=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label('yaw difference (deg)')
    cb.outline.set_visible(False)
    ax.set_xlabel('distance between initial box positions (cm)')
    ax.set_ylabel('RMS joint difference (rad)')
    ax.set_title('Task difference vs. trajectory difference')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return dist, explained, drivers


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('npz', nargs='?', default=os.path.join(REPO, 'data', 'demos.npz'))
    parser.add_argument('--out', default=os.path.join(REPO, 'data', 'analysis'))
    parser.add_argument('--radius', type=float, nargs=2, default=[0.18, 0.25],
                        help='box_radius_range used for collection (m)')
    parser.add_argument('--azimuth', type=float, nargs=2, default=[5.0, 40.0],
                        help='box_azimuth_range used for collection (deg)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.npz)
    names = [str(n) for n in data['joint_names']]
    fps = float(data['fps'])
    episodes = data['episode_index']
    lengths = np.bincount(episodes)
    if np.any(lengths != lengths[0]):
        # Trajectory comparisons need aligned frames; cut to the shortest episode.
        print(f'note: episode lengths differ ({lengths.min()}-{lengths.max()}), '
              f'truncating to {lengths.min()} frames for the joint-space plots')
    n_t = lengths.min()
    rows = [np.flatnonzero(episodes == ep)[:n_t] for ep in range(len(lengths))]
    states = np.stack([data['observation_state'][i] for i in rows])
    actions = np.stack([data['action'][i] for i in rows])
    env0 = np.stack([data['observation_environment_state'][i[0]] for i in rows])

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    style()
    xy, r, az, yaw = task_params(env0)
    plot_task_space(os.path.join(args.out, 'task_space.png'), xy, r, az, yaw,
                    args.radius, args.azimuth, rng)
    dpos, dyaw = plot_coverage(os.path.join(args.out, 'coverage.png'), xy, yaw,
                               args.radius, args.azimuth, rng)
    spread, bounds = plot_joint_space(os.path.join(args.out, 'joint_space.png'),
                                      states, actions, names, fps)
    dist, explained, drivers = plot_episodes(os.path.join(args.out, 'episodes.png'), states, r, az, yaw)

    # --- summary ---
    iu = np.triu_indices(len(xy), 1)
    pair_pos = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    pair_yaw = np.abs((yaw[:, None] - yaw[None] + 45) % 90 - 45)
    np.fill_diagonal(pair_pos, np.inf)
    speed = np.abs(np.diff(states, axis=1)).max(axis=2)
    print(f'{len(xy)} episodes x {n_t} frames at {fps:g} fps = {states.shape[0] * n_t} frames\n')
    print('Task space (box start pose)')
    for label, vals, lim, unit in [('radius', r * 100, np.array(args.radius) * 100, 'cm'),
                                   ('azimuth', az, args.azimuth, 'deg'),
                                   ('yaw mod 90', yaw, (-45, 45), 'deg')]:
        d, p = ks_uniform(vals, *lim)
        print(f'  {label:11s} {vals.min():7.1f} .. {vals.max():6.1f} {unit:3s} '
              f'(range {lim[0]:g} .. {lim[1]:g}, KS vs uniform D={d:.2f} p={p:.2f})')
    nn = pair_pos.min(1) * 100
    print(f'  nearest other demo: median {np.median(nn):.1f} cm, min {nn.min():.1f} cm, '
          f'max {nn.max():.1f} cm')
    close = (pair_pos[iu] < 0.01) & (pair_yaw[iu] < 5)
    print(f'  near-duplicate pairs (< 1 cm and < 5 deg apart): {close.sum()}')
    gaps = dpos.min(1) * 100
    print(f'  random box -> nearest demo position: median {np.median(gaps):.1f} cm, '
          f'90% {np.quantile(gaps, 0.9):.1f} cm, worst {gaps.max():.1f} cm')
    for pt, yt in [(0.01, 5), (0.01, 10), (0.02, 10), (0.02, 45)]:
        print(f'  covered within {pt * 100:g} cm and {yt:g} deg: {covered(dpos, dyaw, pt, yt):.0%}')
    print('\nJoint space')
    for j, n in enumerate(names):
        print(f'  {n:13s} {states[..., j].min():6.2f} .. {states[..., j].max():5.2f} rad, '
              f'mean std across episodes {spread[:, j].mean():.3f} rad')
    print(f'  PCA of whole trajectories: PC1 {explained[0]:.0%}, PC2 {explained[1]:.0%}, '
          f'3 PCs {explained[:3].sum():.1%} of the variance')
    for k, (name, c) in enumerate(drivers):
        print(f'    PC{k + 1} follows the {name} (correlation {c:+.2f})')
    print(f'  RMS joint distance between episodes: median {np.median(dist[iu]):.3f} rad, '
          f'min {dist[iu].min():.3f} rad')
    print(f'  frames with the arm still (< {1e-3 * fps:g} rad/s on every joint): '
          f'{(speed < 1e-3).mean():.0%}')
    for i, (name, _) in enumerate(PHASES):
        a, b = bounds[i], min(bounds[i + 1], n_t)
        if a < n_t:
            print(f'    {name:8s} frames {a:3d}-{b:3d}: mean spread {spread[a:b].mean():.3f} rad')
    print(f'\nFigures in {args.out}')


if __name__ == '__main__':
    main()
