# CompositionalLatentSpace: velocity-only run suite

This overlay adds twenty-nine pressure-free experiments to the
`baskargroup/CompositionalLatentSpace` repository without changing its original
three-channel workflow.

## What "no pressure" means

- Only the FlowBench `u` and `v` solution channels are materialized.
- Every pressure-free configuration sets `model.in_channels: 2`.
- Reconstruction, boundary-layer, same-Re, cross-Re, and full-swap field errors
  are computed only on `u` and `v`.
- Diagnostics retain the regime and geometry probes while excluding pressure
  from all field metrics.
- Figures contain `u` and `v` rows only.

## Runs 1-9

| Run | Experimental change | Key setting |
|---:|---|---|
| 1 | Baseline | `lambda_decorr=0.01` |
| 2 | Strong Pearson decorrelation | `lambda_decorr=0.5` |
| 3 | Intermediate Pearson decorrelation | `lambda_decorr=0.1` |
| 4 | Same-factor invariance | `lambda_inv=0.1` |
| 5 | Same-Re swap consistency | `lambda_swap=0.1` |
| 6 | Cross-Re swap training | `lambda_xswap=0.1` |
| 7 | Static SDF geometry encoder | `static_geometry=true` |
| 8 | Boundary-layer reconstruction | `lambda_bl=1.0` |
| 9 | Every A/B assignment of `[z_mu | z_g | z_xi]` | `lambda_fullswap=0.1` |

Run 9 extends Run 7 with the four missing full-swap assignments. Together with
ordinary reconstruction and cross-Re training, it covers all eight combinations
`AAA`, `AAB`, `ABA`, `ABB`, `BAA`, `BAB`, `BBA`, and `BBB`.

## Runs 10-29

Runs 10-29 are a controlled one-factor-at-a-time search around Run 9. Every
candidate changes exactly one loss weight while preserving the pressure-free
channels, static geometry encoder, architecture, data split, seed, optimizer,
and 200-epoch training schedule.

See [`RUNS10_29_ONE_FACTOR_SWEEP.md`](RUNS10_29_ONE_FACTOR_SWEEP.md) for the
complete matrix and comparison plan.

## Consistent error color scales

Every pressure-free YAML now contains:

```yaml
plotting:
  error_vmax: 0.05
```

The reconstruction, cross-Re transfer, and full-swap absolute-error panels all
use the same `0.00-0.05` colorbar for both `u` and `v` and across every run.
Values above `0.05` saturate at the top of the colormap and are indicated by the
extended colorbar tip. Override the limit manually with:

```bash
python plotting/figures_full_swap_no_pressure.py \
  --config configs/compositional/run10_no_pressure.yaml \
  --checkpoint /path/to/last.ckpt \
  --outdir docs/figures/run10_no_pressure \
  --error-vmax 0.08
```

## Submit individual runs

Each Nova script performs training, standard probes, same-Re and cross-Re
metrics, all-eight-combination diagnostics, and figure generation.

```bash
sbatch --export=ALL,INSTALL_DEPS=0 runNova_run10_no_pressure.sh
```

Use the corresponding script for any run from 1 through 29.

## Submit a selected range of Runs 10-29

The convenience script does not replace the individual shell files. It simply
submits them:

```bash
./submit_runs10_29.sh 10 13
```

Do not run concurrent package installations into the same environment. Prepare
the environment once, then submit with `INSTALL_DEPS=0` as shown above.

## Run-specific outputs

Run `N` writes to:

```text
checkpoints/compositional-runN-no-pressure/
docs/figures/runN_no_pressure/
```

Full-swap runs additionally write:

```text
checkpoints/compositional-runN-no-pressure/<version>/full_swap_metrics.csv
docs/figures/runN_no_pressure/full_swap_reference_grid.png
docs/figures/runN_no_pressure/full_swap_predictions.png
docs/figures/runN_no_pressure/full_swap_errors.png
```

## Compare completed candidates

After several full-swap runs finish, collect their metrics into one CSV:

```bash
python scripts/compare_full_swap_sweep.py --start-run 9 --end-run 29
```

The script sorts by the four new full-swap combinations first, while retaining
cross-Re and probe columns so a low swap ratio cannot hide a damaged latent
representation.

## Verification

From the repository root:

```bash
python tests/test_no_pressure_setup.py
python tests/test_run9_full_swap.py
python tests/test_runs10_29_sweep.py
```

The checks validate the velocity-only data path, Runs 1-9, all eight swap target
mappings, the twenty one-factor configurations, Nova shell syntax, and the
shared plot scale. They do not replace a real FlowBench training run on Nova.
