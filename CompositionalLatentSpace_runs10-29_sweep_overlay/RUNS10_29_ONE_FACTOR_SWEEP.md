# Runs 10-29: one-factor-at-a-time sweep from Run 9

These twenty candidates all start from the pressure-free Run 9 configuration.
Each candidate changes exactly one loss weight and leaves every other setting
unchanged. They are a controlled search, not a guarantee that every run will
improve the result.

## Shared baseline

- Reconstruction = `1.0`
- Regime supervision = `0.1`
- Geometry supervision = `0.1`
- Pearson decorrelation = `0.01`
- Same-factor invariance = `0.1`
- Same-Re swap consistency = `0.1`
- Cross-Re swap = `0.1`
- Full-swap consistency = `0.1`
- Boundary-layer reconstruction = `0.0`
- Static geometry encoder = `True`
- Pressure = excluded
- Plot error colorbar = fixed to `0.00-0.05`

## Candidate matrix

| Run | Only changed setting | Value | Purpose |
|---:|---|---:|---|
| 10 | `lambda_fullswap` | `0.025` | Lower full-swap weight |
| 11 | `lambda_fullswap` | `0.05` | Moderately lower full-swap weight |
| 12 | `lambda_fullswap` | `0.2` | Higher full-swap weight |
| 13 | `lambda_fullswap` | `0.5` | Strong full-swap weight |
| 14 | `lambda_xswap` | `0.05` | Lower cross-re swap weight |
| 15 | `lambda_xswap` | `0.2` | Higher cross-re swap weight |
| 16 | `lambda_swap` | `0.05` | Lower same-re swap weight |
| 17 | `lambda_swap` | `0.2` | Higher same-re swap weight |
| 18 | `lambda_inv` | `0.05` | Lower invariance weight |
| 19 | `lambda_inv` | `0.2` | Higher invariance weight |
| 20 | `lambda_decorr` | `0.0` | Remove pearson decorrelation |
| 21 | `lambda_decorr` | `0.05` | Raise pearson decorrelation |
| 22 | `lambda_regime` | `0.05` | Lower regime supervision |
| 23 | `lambda_regime` | `0.2` | Higher regime supervision |
| 24 | `lambda_geo` | `0.05` | Lower geometry supervision |
| 25 | `lambda_geo` | `0.2` | Higher geometry supervision |
| 26 | `lambda_recon` | `0.5` | Lower reconstruction weight |
| 27 | `lambda_recon` | `2.0` | Higher reconstruction weight |
| 28 | `lambda_bl` | `0.1` | Light boundary-layer weighting |
| 29 | `lambda_bl` | `0.25` | Moderate boundary-layer weighting |

## Recommended comparison order

1. Runs 10-13: tune the new full-swap objective first.
2. Runs 14-19: balance transfer, same-Re interchangeability, and invariance.
3. Runs 20-25: test whether supervision and decorrelation are over- or under-weighted.
4. Runs 26-27: test the reconstruction-to-structure trade-off.
5. Runs 28-29: look for a mild near-wall improvement without repeating Run 8's strong boundary penalty.

## Submission

Each run has its own Nova script. Once dependencies are installed:

```bash
sbatch --export=ALL,INSTALL_DEPS=0 runNova_run10_no_pressure.sh
```

Repeat with run numbers 11 through 29. Every script trains, runs the standard
probes, evaluates all eight latent combinations, and writes both standard and
full-swap figures.

## Selecting the winner

Do not rank candidates from a single transfer image. Compare at least:

- ordinary reconstruction MSE,
- same-Re swap ratio,
- cross-Re swap ratio,
- mean full-swap ratio for `AAB`, `ABA`, `BAB`, and `BBA`,
- `log_re` readability from `z_mu`, `z_g`, and `z_xi`,
- geometry readability from `z_g`,
- whether the fixed `0.05` error scale saturates large regions.

A strong candidate should preserve Run 9's clean `z_mu` and `z_g` separation
while reducing the full-swap ratios toward `1` without materially worsening
reconstruction or cross-Re transfer.

## Automatic comparison after the jobs finish

```bash
python scripts/compare_full_swap_sweep.py \
  --start-run 9 \
  --end-run 29 \
  --output sweep_comparison.csv
```

The table is sorted first by the mean ratio for the four Run-9-specific
combinations (`AAB`, `ABA`, `BAB`, `BBA`) and then by the cross-Re ratio. Keep
reconstruction and probe quality in view before selecting a winner.
