# Running on Nova

Edit `PROJECT_DIR`, `VENV`, and `--mail-user` in `runNova.sh` if your paths differ.

Generate the transient velocity datasets:

```bash
sbatch runNova.sh generate
```

Train after generation finishes:

```bash
sbatch runNova.sh train
```

Or generate missing datasets and train in one job:

```bash
sbatch runNova.sh all
```

Check jobs with:

```bash
squeue -u "$USER"
```

The unsteady model checkpoints are written beneath:

```text
checkpoints/compositional_unsteady_velocity/version_*/
```

The existing `diagnostics/probes.py` is for the steady three-block model and is not run automatically by this shell script.
