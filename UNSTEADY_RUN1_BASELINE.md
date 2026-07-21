# Unsteady velocity-only Run 1 baseline

This configuration is a conceptual unsteady analogue of Run 1 in the paper.

## What matches Run 1

- Reconstruction weight: `1.0`
- Regime supervision: `0.1`
- Geometry supervision: `0.1`
- Pearson cross-block decorrelation: `0.01`
- Geometry is inferred from the flow field (`static_geometry: false`)
- No same-factor invariance, same-Re swap, cross-Re swap, or static geometry encoder

## What differs

- Uses only velocity channels `(u, v)`; pressure is excluded.
- Uses transient sequences.
- Adds a dynamics block `z_eta` and a latent time-stepper `Phi`.
- Adds temporal rollout and one-step latent losses, because an unsteady model cannot learn evolution from the four steady losses alone.

## Run on Nova

Generate missing datasets and train:

```bash
sbatch runNova_run1_unsteady.sh all
```

Generate only:

```bash
sbatch runNova_run1_unsteady.sh generate
```

Train only:

```bash
sbatch runNova_run1_unsteady.sh train
```

Checkpoints are written under:

```text
checkpoints/compositional_unsteady_run1_velocity/version_*/
```
