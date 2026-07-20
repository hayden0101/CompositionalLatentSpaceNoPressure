# Compositional Latent Space for Geometry-Dependent Flow Fields

Starter implementation of the compositional operator-network framework
(see `notes/`) on the **FlowBench 2D lid-driven cavity** dataset — steady/unsteady and unsteady
flow with an object inside the cavity.

## What is implemented (simple version — Path A)

A compositional autoencoder with a block-structured latent

```
z = [ z_mu || z_g || z_xi ]
```

- **z_mu** — regime block: aligned with log Re through a regression head.
- **z_g** — geometry block: aligned with the object shape through a small
  decoder that must reconstruct the SDF from z_g alone.
- **z_xi** — residual block: free capacity for whatever reconstruction
  needs beyond (Re, geometry).

The flow is steady, so there is no dynamics block `z_eta` and no propagator
(they come later with time-dependent cases).

**Losses** (working notes §15): masked L2 reconstruction over the fluid
region (L1), regime supervision on `z_mu`, SDF supervision on `z_g`, and a
Pearson cross-block decorrelation penalty (L6). Weights are set in the config.

**Diagnostics** (working notes §4): `diagnostics/probes.py` fits
cross-validated ridge probes from each block to log Re and simple geometry
descriptors (solid area fraction, centroid). A compositional latent shows
high R² on the matching block and low R² everywhere else.

## Layout

```
main.py                                  # training entry point
configs/compositional/conf.yaml          # all settings
data/dataset.py                          # FlowBench LDC dataset wrapper
models/compositional/networks.py         # encoder / decoder / heads
models/compositional/compositional_ae.py # LightningModule with the loss stack
diagnostics/probes.py                    # linear-probe R^2 diagnostic
notes/                                   # framework PDFs
```

## Data

FlowBench 2D LDC (NS), 512×512 `.npz` tensors:
[LDC_NS_2D on Hugging Face](https://huggingface.co/datasets/BGLab/FlowBench/tree/main/LDC_NS_2D/512x512)

- `x`: `[N, (Re, SDF, mask), 512, 512]`
- `y`: steady `[N, (u, v, p, ...), H, W]` or unsteady `[N, T, C, H, W]` / `[N, C, T, H, W]`. By default only velocity channels `(u, v)` are used, so pressure is excluded.

Set the file paths in `configs/compositional/conf.yaml`.

## Usage

```bash
python3 -m venv sciml && source sciml/bin/activate
pip install -r venv_requirements.txt

# train
python main.py --config configs/compositional/conf.yaml

# diagnose the latent space
python diagnostics/probes.py --checkpoint checkpoints/compositional/<run>/last.ckpt
```

Training logs go to `./logs` (CSV) by default; set `trainer.wandb: true`
for Weights & Biases.

## Next steps (per the project plan)

1. Swap-consistency loss (L12): decode `z_mu` from one sample with `z_g`
   from another and compare against the true cross-combination.
2. Group-structured minibatches (§13) for iVAE-style identifiability.
3. Concept-vector arithmetic diagnostic; INR decoder; HSIC decorrelation.

## Acknowledgments

Repository template from
[Geometry Matters (FlowBench benchmark)](https://arxiv.org/pdf/2501.01453).


## Unsteady velocity-only mode

Use `configs/compositional/unsteady_velocity.yaml` for time-dependent LDC data.
This mode expects consecutive velocity frames, reconstructs only `(u, v)`, adds a
dynamics block `z_eta`, and trains a latent time-stepper `Phi` with the
`lambda_temporal` one-step rollout loss. Pressure is not loaded when
`field_channels: [0, 1]`.

```bash
python main.py --config configs/compositional/unsteady_velocity.yaml
```
