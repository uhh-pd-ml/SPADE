# humite/core/data Structure

This directory handles data loading, preprocessing, and geometry standardization
for calorimeter shower data. It relies heavily on the `awkward` library to
efficiently handle irregular (jagged) hit collections, and converts them into
dense padded `torch.Tensor`s for the model.

The end-to-end flow is:

```
HDF5 file
  -> io.read_shower_file_and_energy        (raw jagged awkward arrays + incident energy)
  -> io.standardize_shower_schema          (map raw fields to a common geometry schema)
  -> sorters.sort_showers                  (define autoregressive hit order)
  -> Adapter.preprocess                    (energy-threshold mask, feature transforms)
  -> BatchCollator                         (pad jagged events -> dense tensors)
  -> Adapter.__call__                      (tokenize, build hierarchical/AR masks)
  -> model
  -> Adapter.decode                        (tokens -> physical bins / energies)
```

`CalorimeterIterableDataset` drives the first four steps per worker; the
`DataLoader`'s `collate_fn` (wired up by `CaloDataModule`) runs the collator and
the adapter's `__call__`.

## Subdirectories and Responsibilities

### `datamodules/`

PyTorch Lightning DataModules that bridge data loading with the training loop.

- **`calo_datamodule.py`**
  - `CaloDataModule`: Main entry point for data handling in training. Builds a
    `BatchCollator` and an `Adapter` (via `build_adapter`) from the
    `DatasetConfig`, and in `setup()` creates train/val/test
    `CalorimeterIterableDataset`s (train uses `repeat=True`). The train/val/test
    DataLoaders share a single `collate_fn` (`_collate_and_adapt`) that runs the
    collator followed by the adapter, and supports separate worker counts,
    `pin_memory`, and `prefetch_factor` per split. Logs a one-time overview of
    the first collated batch's keys/shapes.
  - `DataModuleCfg` + `build_dm_cfg`: Dataclass and helper that extract DataLoader
    knobs (`batch_size`, `num_workers`, `prefetch_factor`, `pin_memory`, and the
    per-split overrides) from a loosely-typed config object (OmegaConf / pydantic /
    plain mapping) by merging its `trainer` and `training` sections.

### `datasets/`

Raw data ingestion and dataset iteration logic.

- **`calorimeter_iterable.py`**
  - `CalorimeterIterableDataset`: A streaming `IterableDataset`. It expands file
    glob patterns per type, shards files/rows across DDP ranks and DataLoader
    workers, loads showers via `io.py`, standardizes the schema, sorts hits, runs
    `adapter.preprocess`, and yields one per-event dict at a time. Each event dict
    holds `energy`, `incident_energy`, `n_hits`, `hit_mask`, `spatial_axis_names`,
    and one entry per spatial axis (e.g. `x`/`y`/`z`).
  - Supports multiple sharding strategies: per-file sharding across ranks/workers,
    and **row-level sharding** (`worker_row_sharding`, plus rank-level row
    sharding when there are fewer files than ranks) with `block` or `interleaved`
    modes. Other options include file/data shuffling, `n_files_at_once` batching,
    an optional RAM cache (`use_ram_cache`), and `load_only_once` to cache a
    rank's events for repeated epochs.
- **`io.py`**
  - Low-level I/O utilities. All functions return raw arrays (Awkward / NumPy)
    without preprocessing or tokenization.
  - `read_shower_file_and_energy`: Reads HDF5 files into `awkward.Array`s plus a
    flat incident-energy array. Auto-detects the file layout and dispatches to the
    matching loader:
    - `showers` dataset of shape `(N, hits, 4)` (dense cartesian `x,y,z,energy`,
      energy from `genE`/`incident_energies`) -> `_load_cartesian_showers`.
    - `events` (`(N, 5, 3700)`) + `n_points` with left padding -> `_load_events_showers`.
    - separate `energy`/`x`/`y`/`z` datasets with energy>0 masking -> `_load_separated_showers`.
  - Supports row selection via `n_load`, `start`/`stop`/`step` slicing, or a
    seeded random permutation (`_make_row_selector`); the two are mutually
    exclusive.
  - `get_h5_row_count`: Returns the number of showers in a file (used for row
    sharding) across the supported layouts.
  - `standardize_shower_schema`: Dispatches to the registered geometry
    standardizer for the given `dataset_type`.

### `geometry/`

Definitions and standardizers for different detector geometries.

- **`__init__.py`**: Registry mapping `dataset_type` to a standardizer
  (`GEOMETRY_STANDARDIZERS`), to its spatial field names
  (`GEOMETRY_SPATIAL_FIELDS`), and to optional static per-axis codebook sizes
  (`GEOMETRY_SPATIAL_CODEBOOK_SIZES`). Exposes `get_spatial_field_names` and
  `get_spatial_codebook_sizes`. Currently only `ECAL` (spatial fields
  `("x", "y", "z")`) is registered; ECAL takes its per-axis bin counts from the
  dataset config (`nbins_*`), so no static codebook is needed.
- **`ecal_cartesian.py`**
  - `standardize_ecal_cartesian`: ECAL data already arrives as `(x, y, z, energy)`,
    so this is a pass-through.

### `preprocessing/`

Transformation logic to convert physical data into model-ready tensors and back.

- **`adapters.py`**
  - `BaseAdapter` and subclasses (`SplitAdapter`, `CombinedAdapter`,
    `HSplitAdapter`, `HCombinedAdapter`), selected by `build_adapter(dataset)`
    from `dataset.axis_mode` (`split`/`combined`) and `dataset.hierarchical`.
  - **`preprocess`**: Applies the per-hit energy threshold to build `hit_mask`
    and `n_hits`, then applies per-feature transforms (`apply_feature_transform`)
    to each spatial axis and to energy. Runs in the dataset workers, before
    collation.
  - **Tokenization** (`__call__`): Converts continuous bin coordinates into
    per-axis integer tokens (`*_token`, with reserved `PAD=0` and per-axis
    `MISSING=nbins+1`). Energy is kept continuous and, if
    `use_energy_tokenization` is set, additionally tokenized — either with
    **uniform** binning (optionally in log space) or **`hcal_resolution`**
    binning whose edges follow `sigma_E(E)` (see `build_hcal_resolution_bin_edges`).
    - `SplitAdapter` keeps one token stream per axis.
    - `CombinedAdapter` packs the per-axis tokens into a single `token_ids`
      stream (mixed-radix over `axis_nbins`).
  - **Hierarchical / AR scheduling**: `HSplitAdapter` and `HCombinedAdapter`
    shift each stream onto a staggered autoregressive timeline (e.g. split:
    `z@1, x@2, y@3, energy@4`; combined: `token_ids@1, energy@2`), filling the
    gaps with `MISSING` ids and producing the `attention_mask` and per-axis
    `axis_valid_mask`.
  - **Decoding** (`decode`): Inverse of the above. Realigns the staggered streams
    back to physical time, decomposes combined tokens, removes PAD/MISSING,
    intersects with stop/attention masks during generation, and returns
    `hit_features` (`x,y,z` as bin integers + physical energy) and a `hit_mask`
    of valid physical hits. `build_ar_aligned_z_tokens` builds a ground-truth,
    AR-aligned `z` token tensor for evaluation/conditioning.
- **`collators.py`**
  - `BatchCollator`: Collates variable-length per-event dicts into fixed-size
    padded tensors at `dataset.pad_length`. Emits `hit_features`
    (`[B, L, len(axes)+1]`, the spatial axes followed by energy), `hit_mask`,
    `incident_energy` (+ transformed `incident_energy_cond`), `n_hits`
    (clamped to `pad_length`, + transformed `n_hits_cond`), and
    `spatial_axis_names`.
- **`global_features.py`**
  - `compute_global_features`: Single source of truth for global shower
    observables computed directly from a point cloud — totals and hit counts,
    energy/center-of-gravity, per-axis and radial widths (`sigma_*`),
    `E_max/E_total`, depth of shower max, `z_span`, radial containment radii
    (`R50`/`R90`), optional `n_hits_E>X` counts, optional incident-energy ratios,
    and optional per-layer profiles (`E_layer`, `sigma_r_layer`). Used for
    evaluation/conditioning rather than in the core token pipeline.
- **`sorters.py`**
  - `sort_showers`: Sorts hits within each shower to define the autoregressive
    sequence order. Supports `energy`, `layer`, `layer_energy`,
    `layer_desc_energy`, `layer_random`, `random`, `layer_x`, `radius_from_ip`,
    `layer_radius_from_ip`, and `none`. Longitudinal/reference axes are derived
    from the geometry's spatial field names.
- **`utils.py`**
  - Awkward/NumPy helpers and feature-transform logic: `ak_pad` (jagged ->
    padded, optional mask), `ak_to_np_stack` (fast field-stacking to NumPy),
    `get_valid_energy_mask`, `apply_feature_transform` (affine + optional
    forward/inverse func, dispatching over torch/NumPy/awkward backends),
    `build_hcal_resolution_bin_edges` / `bin_centers_from_edges` (energy bin
    construction), `decompose_combined_spatial_tokens` (inverse of the combined
    packing), and `summarize_value` (batch logging helper).

## Summary of Key Classes / Functions

| Class / Function                     | Responsibility                                                         |
| :----------------------------------- | :--------------------------------------------------------------------- |
| `CaloDataModule`                     | **Orchestrator**: builds collator + adapter, sets up datasets/loaders. |
| `CalorimeterIterableDataset`         | **Streaming**: shards, loads, sorts, preprocesses, yields events.      |
| `io.read_shower_file_and_energy`     | **I/O**: raw HDF5 (3 layouts) -> jagged `awkward.Array` + incident E.  |
| `standardize_shower_schema`          | **Schema**: map raw fields to a common geometry schema.                |
| `sort_showers`                       | **Ordering**: define the autoregressive hit sequence.                  |
| `BatchCollator`                      | **Batching**: pad jagged events -> dense `torch.Tensor`s.              |
| `SplitAdapter` / `CombinedAdapter`   | **Transform**: physical values \<-> per-axis / packed tokens.          |
| `HSplitAdapter` / `HCombinedAdapter` | **Transform (AR)**: hierarchical, staggered autoregressive scheduling. |
| `compute_global_features`            | **Observables**: global shower features from a point cloud.            |
