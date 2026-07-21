# Run 1, steady-state, no-pressure setup

This setup reproduces the **training logic of Run 1** from the paper while
reconstructing only the velocity channels `(u, v)`. It is therefore a
**no-pressure Run-1 analogue**, not a numerically identical replication of the
paper's three-channel `(u, v, p)` experiment.

## What Run 1 uses

- FlowBench LDC easy split: 2,400 train / 600 test
- Resolution: 256 x 256
- Batch size: 16
- Epochs: 200
- Latent blocks: `z_mu=4`, `z_g=32`, `z_xi=16`
- Geometry inferred from the flow (`static_geometry: false`)
- Loss weights:
  - reconstruction: 1.0
  - regime supervision: 0.1
  - geometry supervision: 0.1
  - Pearson decorrelation: 0.01
- Disabled: invariance, same-Re swap, cross-Re swap, boundary loss

## Apply to your repository

From the repository root:

```bash
cp configs/compositional/run1_no_pressure.yaml configs/compositional/
cp runNova_run1_no_pressure.sh .
git apply patches/run1_no_pressure.patch
```

If you extracted this setup beside the repository, copy the three files from
this package into the matching locations first.

Verify the channel change:

```bash
grep -n "y\[:, :2\]" data/dataset.py
```

## Submit on Nova

```bash
sbatch runNova_run1_no_pressure.sh
```

Check the job:

```bash
squeue -u "$USER"
```

The experiment writes checkpoints under:

```text
checkpoints/compositional_no_pressure_run1/version_*/
```

The shell script automatically runs `diagnostics/probes.py` on the newest
`last.ckpt` after training.

## Expected scientific comparison

The paper's Run 1 uses `(u, v, p)`, so your absolute reconstruction MSE and
probe scores do not need to match its table exactly. The correct comparison is
qualitative: high `log Re` readability from `z_mu`, geometry readability from
`z_g`, and likely Reynolds-number leakage into `z_g` and `z_xi` because Run 1
uses only the weak Pearson decorrelation penalty.
