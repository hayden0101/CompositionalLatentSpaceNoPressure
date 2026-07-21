"""Generate transient, velocity-only lid-driven-cavity sequences.

This utility reuses the geometry/Reynolds-number inputs from a FlowBench x-file
and advances a simple finite-difference incompressible Navier-Stokes solver.
Only u and v are saved. Pressure is an internal projection variable and is not
part of the generated dataset or learning target.

The solver is intended as a reproducible research-data generator and should be
validated (grid/time-step refinement and comparison against a trusted CFD
solver) before publication-quality use.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def laplacian(q: np.ndarray, dx: float) -> np.ndarray:
    out = np.zeros_like(q)
    out[1:-1, 1:-1] = (
        q[1:-1, 2:] + q[1:-1, :-2] + q[2:, 1:-1] + q[:-2, 1:-1]
        - 4.0 * q[1:-1, 1:-1]
    ) / (dx * dx)
    return out


def derivatives(q: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    qx = np.zeros_like(q)
    qy = np.zeros_like(q)
    qx[1:-1, 1:-1] = (q[1:-1, 2:] - q[1:-1, :-2]) / (2.0 * dx)
    qy[1:-1, 1:-1] = (q[2:, 1:-1] - q[:-2, 1:-1]) / (2.0 * dx)
    return qx, qy


def apply_velocity_bc(u: np.ndarray, v: np.ndarray, fluid: np.ndarray, lid_speed: float) -> None:
    # No-slip side and bottom walls; moving top lid.
    u[0, :] = 0.0
    v[0, :] = 0.0
    u[-1, :] = lid_speed
    v[-1, :] = 0.0
    u[:, 0] = 0.0
    v[:, 0] = 0.0
    u[:, -1] = 0.0
    v[:, -1] = 0.0
    # Solid obstacle is stationary.
    solid = fluid < 0.5
    u[solid] = 0.0
    v[solid] = 0.0


def pressure_poisson(p: np.ndarray, rhs: np.ndarray, dx: float, iters: int) -> np.ndarray:
    pn = np.empty_like(p)
    for _ in range(iters):
        pn[...] = p
        p[1:-1, 1:-1] = 0.25 * (
            pn[1:-1, 2:] + pn[1:-1, :-2] + pn[2:, 1:-1] + pn[:-2, 1:-1]
            - dx * dx * rhs[1:-1, 1:-1]
        )
        # Homogeneous Neumann pressure BCs and one reference point.
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        p[0, 0] = 0.0
    return p


def simulate(fluid: np.ndarray, re: float, n_steps: int, save_every: int,
             dt: float, lid_speed: float, poisson_iters: int) -> np.ndarray:
    n = fluid.shape[0]
    dx = 2.0 / (n - 1)
    nu = lid_speed * 2.0 / max(float(re), 1.0)

    u = np.zeros((n, n), dtype=np.float32)
    v = np.zeros((n, n), dtype=np.float32)
    p = np.zeros((n, n), dtype=np.float32)
    apply_velocity_bc(u, v, fluid, lid_speed)

    frames = [np.stack([u, v], axis=0)]
    for step in range(1, n_steps + 1):
        ux, uy = derivatives(u, dx)
        vx, vy = derivatives(v, dx)
        u_star = u + dt * (-(u * ux + v * uy) + nu * laplacian(u, dx))
        v_star = v + dt * (-(u * vx + v * vy) + nu * laplacian(v, dx))
        apply_velocity_bc(u_star, v_star, fluid, lid_speed)

        usx, _ = derivatives(u_star, dx)
        _, vsy = derivatives(v_star, dx)
        rhs = (usx + vsy) / dt
        p = pressure_poisson(p, rhs, dx, poisson_iters)
        px, py = derivatives(p, dx)
        u = u_star - dt * px
        v = v_star - dt * py
        apply_velocity_bc(u, v, fluid, lid_speed)

        if step % save_every == 0:
            frames.append(np.stack([u, v], axis=0).astype(np.float32))
    return np.stack(frames, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-x', required=True, help='FlowBench x .npz file')
    parser.add_argument('--output', required=True, help='Output transient .npz file')
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=20)
    parser.add_argument('--dt', type=float, default=2e-4)
    parser.add_argument('--lid-speed', type=float, default=1.0)
    parser.add_argument('--poisson-iters', type=int, default=40)
    args = parser.parse_args()

    raw = np.load(args.input_x)['data']
    if args.num_samples is not None:
        raw = raw[:args.num_samples]

    try:
        from scipy.ndimage import zoom
    except ImportError as exc:
        raise SystemExit('scipy is required for dataset generation') from exc

    sequences, sdf_out, mask_out, re_out = [], [], [], []
    for i, sample in enumerate(raw):
        re = float(sample[0, 0, 0])
        sdf = sample[1].astype(np.float32)
        mask = sample[2].astype(np.float32)
        if mask.max() > 1.0:
            mask /= 255.0
        if sdf.shape[0] != args.resolution:
            scale = args.resolution / sdf.shape[0]
            sdf = zoom(sdf, scale, order=1).astype(np.float32)
            mask = zoom(mask, scale, order=0).astype(np.float32)
        seq = simulate(mask, re, args.steps, args.save_every, args.dt,
                       args.lid_speed, args.poisson_iters)
        sequences.append(seq)
        sdf_out.append(sdf[None])
        mask_out.append(mask[None])
        re_out.append(re)
        print(f'[{i + 1}/{len(raw)}] Re={re:g}, frames={seq.shape[0]}')

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        velocity=np.stack(sequences).astype(np.float32),  # [N,T,2,H,W]
        sdf=np.stack(sdf_out).astype(np.float32),          # [N,1,H,W]
        mask=np.stack(mask_out).astype(np.float32),        # [N,1,H,W]
        re=np.asarray(re_out, dtype=np.float32),            # [N]
        dt=np.float32(args.dt * args.save_every),
    )
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
