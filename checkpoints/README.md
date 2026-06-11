# Checkpoints

To keep the repository small, the pretrained checkpoints are hosted separately.
Download and unpack them into this directory with:

```sh
# run this in the `checkpoints/` directory
./download_checkpoints.sh
# use this command to skip the x16_combined setting as it 900MB and bigger as all other 9 checkpoints combined.
SKIP_X16_COMBINED=1 ./download_checkpoints.sh
```

## Layout

Each checkpoint ships in its own folder alongside a minimal, self-contained
`config.yaml`. `humite-generate` loads that config automatically, so the
architecture always matches the weights and you never pass overrides.

```
checkpoints/
├── spade/                  # SPADE model (axis_mode: split) — paper's main model
│   ├── getting_square_x1/   { model.ckpt, config.yaml }
│   ├── getting_square_x4/    …
│   ├── getting_square_x16/   …
│   ├── getting_square_x1_merged/ …
│   └── getting_high/         …
└── combined/               # Combined baseline (axis_mode: combined)
    └── … (same five variants)
```

`model.ckpt` is the inference checkpoint. Note the size contrast — SPADE's per-axis vocabulary stays ~12 MB at
every granularity, while the Combined baseline's single joint vocabulary grows
cubically (≈0.9 GB at x16); this is the scaling argument behind SPADE.

## Generating from a checkpoint

The command is identical for every checkpoint — the bundled config sets the
architecture (`axis_mode`, binning), sampling, and any postprocessing:

```sh
humite-generate +generate.checkpoint=checkpoints/spade/getting_high/model.ckpt
```

```sh
# any variant, any model — just point at its model.ckpt:
humite-generate +generate.checkpoint=checkpoints/combined/getting_square_x16/model.ckpt
humite-generate +generate.checkpoint=checkpoints/spade/getting_square_x1_merged/model.ckpt
```

By default this **writes 1000 showers** (energies uniform in 10–100 GeV) to
`generated_showers.h5` in the directory you run the command from (the run does
*not* `chdir` into a timestamped output dir). Override the count and destination
with more `+generate.*` flags:

```sh
humite-generate +generate.checkpoint=checkpoints/spade/getting_high/model.ckpt \
  +generate.n_showers=5000 \
  +generate.output=showers/spade_high.h5      # parent dirs are created automatically
```

| flag                                                      | default                | meaning                                              |
| --------------------------------------------------------- | ---------------------- | ---------------------------------------------------- |
| `+generate.n_showers=`                                    | `1000`                 | number of showers to generate                        |
| `+generate.output=`                                       | `generated_showers.h5` | output path (relative to your cwd, or absolute)      |
| `+generate.energy_gev=`                                   | —                      | fixed incident energy in GeV; omit to sample a range |
| `+generate.energy_min_gev=` / `+generate.energy_max_gev=` | `10` / `100`           | incident-energy range (GeV) when no fixed energy     |

The HDF5 holds `hit_features` (n, max_hits, 4=[x, y, z, E]), a `hit_mask`
flagging real hits, and `incident_energy` (n,) in GeV.

For interactive use, the Python API `generate_showers` runs the same code path
but returns these arrays in memory instead — the example notebook
[`examples/generate_and_plot.ipynb`](../examples/generate_and_plot.ipynb) walks
through that.

## Postprocessing (energy-sum filter)

The checkpoints bundle the energy-sum lower-envelope filter
described in the paper (App. B Postprocessing): generated showers whose total
deposited energy falls below `s·(a·E_inc^b + c)` are regenerated under the same
conditioning. The coefficients live in each config's
`trainer.generative_eval.sampling` block and are applied automatically. They are
fit per dataset and must be in the same energy units as the decoded hits.
