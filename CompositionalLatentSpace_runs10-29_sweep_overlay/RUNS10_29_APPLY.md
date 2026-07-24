# Apply Runs 10-29 and consistent plot scales

This update assumes the pressure-free Runs 1-9 files are already present.

## Git patch method

From the repository root:

```bash
git apply --check /path/to/CompositionalLatentSpace_runs10-29_sweep.patch
git apply /path/to/CompositionalLatentSpace_runs10-29_sweep.patch
chmod +x runNova_run{10..29}_no_pressure.sh submit_runs10_29.sh \
  scripts/compare_full_swap_sweep.py tests/test_runs10_29_sweep.py
python tests/test_runs10_29_sweep.py
```

## Overlay ZIP method

Extract the incremental overlay directly into the repository root and allow it
to merge with the existing folders. Then run the same `chmod` and test commands
above.

## First candidate

A cautious first mini-sweep is Runs 10-13, which changes only the new full-swap
weight:

```bash
./submit_runs10_29.sh 10 13
```

Use `INSTALL_DEPS=0` only after the shared virtual environment is prepared.

## Plot scale

All pressure-free configurations now use the same absolute-error colorbar:

```yaml
plotting:
  error_vmax: 0.05
```

This makes the `u` and `v` error maps and different runs directly comparable.
Errors above `0.05` saturate rather than silently expanding the colorbar.
