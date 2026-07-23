# CompositionalLatentSpace: velocity-only run suite

This overlay adds eight pressure-free experiments to the current
`baskargroup/CompositionalLatentSpace` repository without changing its original
three-channel workflow.

## What "no pressure" means

- `NoPressureLDCDataset` materializes only the first two FlowBench solution
  channels, `u` and `v`, in a new contiguous tensor.
- Every no-pressure config sets `model.in_channels: 2`, so the encoder reads two
  channels and the decoder predicts two channels.
- Reconstruction, boundary-layer, same-Re swap, and cross-Re swap losses are
  therefore computed only on `u` and `v`.
- The diagnostics use the same upstream probe and swap implementation, but their
  field errors are evaluated only on `u` and `v`.
- The plotting script creates two-row figures for `u` and `v`; no pressure panel
  is produced.

The wrappers delegate to the repository's existing training and diagnostic
entry points. This keeps the original batching, loss implementation, logging,
and checkpoint behavior intact while replacing the dataset with the velocity-
only adapter.

## Eight configurations

| Run | Change from the preceding experiment | Key settings |
|---|---|---|
| 1 | Baseline | `lambda_decorr=0.01` |
| 2 | Strong Pearson decorrelation | `lambda_decorr=0.5` |
| 3 | Intermediate Pearson decorrelation | `lambda_decorr=0.1` |
| 4 | Same-factor invariance, L10 | `lambda_inv=0.1` |
| 5 | Same-Re swap consistency, L12 | `lambda_swap=0.1` |
| 6 | Cross-Re swap training | `lambda_xswap=0.1` |
| 7 | Static SDF geometry encoder | `static_geometry=true` |
| 8 | Boundary-layer reconstruction, L3 | `lambda_bl=1.0` |

All runs retain `lambda_recon=1.0`, `lambda_regime=0.1`,
`lambda_geo=0.1`, 200 epochs, batch size 16, resolution 256, seed 0, and the
same latent dimensions as the upstream experiments.

## Apply the overlay

From outside a fresh clone:

```bash
git clone https://github.com/baskargroup/CompositionalLatentSpace.git
unzip -o CompositionalLatentSpace_no_pressure_overlay.zip \
  -d CompositionalLatentSpace
cd CompositionalLatentSpace
chmod +x runNova_run*_no_pressure.sh main_no_pressure.py \
  diagnostics/probes_no_pressure.py plotting/figures_no_pressure.py \
  scripts/latest_checkpoint.py
```

The alternative patch can be applied from the repository root:

```bash
git apply CompositionalLatentSpace_no_pressure.patch
```

## Submit a run on Nova

Each Slurm script performs all three stages in order:

1. Train the selected configuration.
2. Find the newest `last.ckpt` in that run's dedicated checkpoint directory and
   run the probe and swap diagnostics.
3. Generate reconstruction and cross-Re transfer figures as PNG and PDF files.

Example:

```bash
sbatch runNova_run1_no_pressure.sh
```

The remaining jobs are submitted the same way:

```bash
sbatch runNova_run2_no_pressure.sh
sbatch runNova_run3_no_pressure.sh
sbatch runNova_run4_no_pressure.sh
sbatch runNova_run5_no_pressure.sh
sbatch runNova_run6_no_pressure.sh
sbatch runNova_run7_no_pressure.sh
sbatch runNova_run8_no_pressure.sh
```

## Environment and path overrides

The scripts preserve the supplied virtual-environment path:

```text
/work/mech-ai/haydenc1/packages/CompositionalLatentSpace/sciml
```

Override it at submission time when needed:

```bash
sbatch --export=ALL,VENV=/path/to/sciml runNova_run1_no_pressure.sh
```

Dependency installation is enabled by default to match the supplied Nova
script. Once the environment is prepared, skip repeated installation with:

```bash
sbatch --export=ALL,INSTALL_DEPS=0 runNova_run1_no_pressure.sh
```

The YAML files retain the upstream Nova data paths. Edit the four `file_path_*`
entries if the FlowBench files live elsewhere.

## Outputs

Run 1, for example, writes checkpoints under:

```text
checkpoints/compositional-run1-no-pressure/
```

and figures under:

```text
docs/figures/run1_no_pressure/
```

Every other run uses its own checkpoint and figure directories, preventing the
runs from stepping on one another's results.

## Quick static verification

From the repository root after applying the overlay:

```bash
python tests/test_no_pressure_setup.py
```

This checks all eight loss configurations, confirms two input/output channels,
and verifies that every Nova script invokes training, diagnostics, and plotting.
It does not replace a real data-and-GPU smoke test on Nova.
