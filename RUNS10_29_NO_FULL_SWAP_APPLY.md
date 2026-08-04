# Apply Runs 10-29 without full swap

This package assumes the pressure-free Runs 1-9 overlay is already present.
Run 9 may remain in the repository, but Runs 10-29 do not inherit or invoke its
full-swap implementation.

## Clean application

From the repository root:

```bash
git apply --check /path/to/CompositionalLatentSpace_runs10-29_no_full_swap.patch
git apply /path/to/CompositionalLatentSpace_runs10-29_no_full_swap.patch
chmod +x runNova_run{10..29}_no_pressure.sh submit_runs10_29_no_full_swap.sh
python tests/test_runs10_29_no_full_swap.py
```

Submit a small first batch:

```bash
./submit_runs10_29_no_full_swap.sh 10 13
```

Or submit one run:

```bash
sbatch --export=ALL,INSTALL_DEPS=0 runNova_run10_no_pressure.sh
```

Do not apply the earlier full-swap-based Runs 10-29 patch on top of this one.
A separate replacement patch is supplied for repositories where that earlier
sweep was already applied.
