# Runs 10-29: standard-swap one-factor sweep

These twenty pressure-free candidates use **Run 7** as the baseline. Each
changes exactly one value. None uses the Run 9 full-swap loss, full-swap
model, eight-combination diagnostic, or full-swap matrix plot.

The unchanged baseline is:

- Reconstruction = `1.0`
- Regime supervision = `0.1`
- Geometry supervision = `0.1`
- Pearson decorrelation = `0.01`
- Same-factor invariance = `0.1`
- Same-Re swap consistency = `0.1`
- Cross-Re swap = `0.1`
- Boundary-layer reconstruction = `0.0`
- Static geometry encoder = `True`
- Pressure included = `False`

## Candidate runs

### Run 10

- **Changed setting:** Cross-Re swap = `0.025`
- **Purpose:** Lower cross-Re swap weight.
- All other Run 7 settings remain unchanged.

### Run 11

- **Changed setting:** Cross-Re swap = `0.05`
- **Purpose:** Moderately lower cross-Re swap weight.
- All other Run 7 settings remain unchanged.

### Run 12

- **Changed setting:** Cross-Re swap = `0.2`
- **Purpose:** Higher cross-Re swap weight.
- All other Run 7 settings remain unchanged.

### Run 13

- **Changed setting:** Cross-Re swap = `0.5`
- **Purpose:** Strong cross-Re swap weight.
- All other Run 7 settings remain unchanged.

### Run 14

- **Changed setting:** Same-Re swap consistency = `0.025`
- **Purpose:** Lower same-Re swap weight.
- All other Run 7 settings remain unchanged.

### Run 15

- **Changed setting:** Same-Re swap consistency = `0.05`
- **Purpose:** Moderately lower same-Re swap weight.
- All other Run 7 settings remain unchanged.

### Run 16

- **Changed setting:** Same-Re swap consistency = `0.2`
- **Purpose:** Higher same-Re swap weight.
- All other Run 7 settings remain unchanged.

### Run 17

- **Changed setting:** Same-factor invariance = `0.025`
- **Purpose:** Lower same-factor invariance weight.
- All other Run 7 settings remain unchanged.

### Run 18

- **Changed setting:** Same-factor invariance = `0.05`
- **Purpose:** Moderately lower same-factor invariance weight.
- All other Run 7 settings remain unchanged.

### Run 19

- **Changed setting:** Same-factor invariance = `0.2`
- **Purpose:** Higher same-factor invariance weight.
- All other Run 7 settings remain unchanged.

### Run 20

- **Changed setting:** Pearson decorrelation = `0.0`
- **Purpose:** Remove Pearson decorrelation.
- All other Run 7 settings remain unchanged.

### Run 21

- **Changed setting:** Pearson decorrelation = `0.05`
- **Purpose:** Increase Pearson decorrelation.
- All other Run 7 settings remain unchanged.

### Run 22

- **Changed setting:** Pearson decorrelation = `0.1`
- **Purpose:** Stronger Pearson decorrelation.
- All other Run 7 settings remain unchanged.

### Run 23

- **Changed setting:** Regime supervision = `0.05`
- **Purpose:** Lower regime-supervision weight.
- All other Run 7 settings remain unchanged.

### Run 24

- **Changed setting:** Regime supervision = `0.2`
- **Purpose:** Higher regime-supervision weight.
- All other Run 7 settings remain unchanged.

### Run 25

- **Changed setting:** Geometry supervision = `0.05`
- **Purpose:** Lower geometry-supervision weight.
- All other Run 7 settings remain unchanged.

### Run 26

- **Changed setting:** Geometry supervision = `0.2`
- **Purpose:** Higher geometry-supervision weight.
- All other Run 7 settings remain unchanged.

### Run 27

- **Changed setting:** Reconstruction = `0.5`
- **Purpose:** Lower reconstruction weight.
- All other Run 7 settings remain unchanged.

### Run 28

- **Changed setting:** Reconstruction = `2.0`
- **Purpose:** Higher reconstruction weight.
- All other Run 7 settings remain unchanged.

### Run 29

- **Changed setting:** Boundary-layer reconstruction = `0.1`
- **Purpose:** Add a light boundary-layer reconstruction loss.
- All other Run 7 settings remain unchanged.

## Consistent figure scales

Every standard reconstruction and cross-Re transfer error map uses:

- `u` absolute-error colorbar = `0.00` to `0.05`
- `v` absolute-error colorbar = `0.00` to `0.05`
- The same limits across every run

Errors above `0.05` saturate at the top of the colorbar. Override the limit
with `--error-vmax`, or change `plotting.error_vmax` in a YAML file.
