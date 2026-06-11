"""
Low-level I/O utilities for reading calorimeter shower datasets.
All functions here return raw arrays (Awkward, NumPy, etc.),
without any preprocessing or tokenization.
"""

from __future__ import annotations

import logging
from typing import Union, cast

import awkward as ak
import h5py
import numpy as np
import vector

from ..geometry import GEOMETRY_STANDARDIZERS

logger = logging.getLogger(__name__)

vector.register_awkward()


RowSelector = Union[slice, np.ndarray]


def get_h5_row_count(filepath: str) -> int:
    with h5py.File(filepath, "r") as h5file:
        if "showers" in h5file:
            ds_showers = cast(h5py.Dataset, h5file["showers"])  # type: ignore[index]
            return int(ds_showers.shape[0])
        if "events" in h5file and "n_points" in h5file:
            ds_events = cast(h5py.Dataset, h5file["events"])  # type: ignore[index]
            return int(ds_events.shape[0])
        if "energy" in h5file and "x" in h5file and "y" in h5file and "z" in h5file:
            ds_energy = cast(h5py.Dataset, h5file["energy"])  # type: ignore[index]
            return int(ds_energy.shape[0])
    raise KeyError("Expected 'showers', 'events' or ('energy', 'x', 'y', 'z') datasets.")


def _make_row_selector(
    total_rows: int,
    n_load: int | None,
    start: int | None,
    stop: int | None,
    step: int | None,
    random_seed: int | None,
) -> RowSelector:
    start_idx = 0 if start is None else max(0, int(start))
    stop_idx = int(total_rows) if stop is None else min(int(total_rows), max(0, int(stop)))
    step_size = 1 if step is None else max(1, int(step))

    if start_idx > stop_idx:
        start_idx = stop_idx

    if random_seed is not None and (
        start is not None or stop is not None or step not in (None, 1)
    ):
        raise ValueError("random_seed cannot be combined with start/stop/step row slicing")

    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        permutation = rng.permutation(int(total_rows))
        take = int(total_rows) if n_load is None else min(int(n_load), int(total_rows))
        selected = np.sort(permutation[:take])
        return selected.astype(np.int64)

    selected_count = len(range(start_idx, stop_idx, step_size))
    if n_load is not None:
        selected_count = min(selected_count, max(0, int(n_load)))

    if step_size == 1:
        return slice(start_idx, start_idx + selected_count)

    stop_for_stride = start_idx + step_size * selected_count
    return np.arange(start_idx, stop_for_stride, step_size, dtype=np.int64)


def _selector_count(selector: RowSelector) -> int:
    if isinstance(selector, slice):
        start = 0 if selector.start is None else int(selector.start)
        stop = start if selector.stop is None else int(selector.stop)
        step = 1 if selector.step is None else int(selector.step)
        return len(range(start, stop, step))
    return int(np.asarray(selector).shape[0])


def read_shower_file_and_energy(
    filepath,
    n_load=None,
    chunk_size=1000,
    random_seed=None,
    start=None,
    stop=None,
    step=None,
    dataset_type=None,
):
    """Load calorimeter data, automatically handling cartesian and cylindrical layouts."""

    with h5py.File(filepath, "r") as h5file:
        if "showers" in h5file:
            ds_showers = cast(h5py.Dataset, h5file["showers"])  # type: ignore[index]
            total_showers = int(ds_showers.shape[0])

            if "genE" in h5file:
                ds_energy = cast(h5py.Dataset, h5file["genE"])  # type: ignore[index]
            elif "incident_energies" in h5file:
                ds_energy = cast(h5py.Dataset, h5file["incident_energies"])  # type: ignore[index]
            else:
                raise KeyError(
                    "Energy dataset not found (expected 'genE' or 'incident_energies')."
                )
            row_selector = _make_row_selector(
                total_showers,
                n_load=n_load,
                start=start,
                stop=stop,
                step=step,
                random_seed=random_seed,
            )
            dataset_showers = np.asarray(ds_showers[row_selector])
            raw_energy = np.asarray(ds_energy[row_selector])

            n_selected = _selector_count(row_selector)
            if n_load is not None and start is None and stop is None and step in (None, 1):
                logger.info(f"Loading {n_selected} showers (available: {total_showers}).")

            incident_energy = _flatten_incident_energy(raw_energy)

            if dataset_showers.ndim == 3:
                return _load_cartesian_showers(
                    dataset_showers, incident_energy, n_load, chunk_size
                )

            raise ValueError("Unsupported shower tensor shape. Expected (N, hits, 4).")

        elif "events" in h5file and "n_points" in h5file:
            return _load_events_showers(
                h5file,
                n_load=n_load,
                random_seed=random_seed,
                start=start,
                stop=stop,
                step=step,
            )

        elif all(k in h5file for k in ["energy", "x", "y", "z"]):
            return _load_separated_showers(
                h5file,
                n_load=n_load,
                random_seed=random_seed,
                start=start,
                stop=stop,
                step=step,
            )

        else:
            raise KeyError("Expected 'showers', 'events' or ('energy', 'x', 'y', 'z') datasets.")


def _load_events_showers(h5file, n_load, random_seed, start=None, stop=None, step=None):
    ds_events = h5file["events"]
    ds_npoints = h5file["n_points"]

    # incident energy is 'energy' in this format
    if "energy" in h5file:
        raw_incident_energy = np.asarray(h5file["energy"][()])
    else:
        raise KeyError("Incident energy not found (expected 'energy').")

    total_showers = int(ds_events.shape[0])
    row_selector = _make_row_selector(
        total_showers, n_load=n_load, start=start, stop=stop, step=step, random_seed=random_seed
    )
    if isinstance(row_selector, slice):
        indices = np.arange(total_showers, dtype=np.int64)[row_selector]
    else:
        indices = np.asarray(row_selector, dtype=np.int64)
    indices.sort()

    # Load data
    events = ds_events[indices]
    n_points = ds_npoints[indices]
    incident_energy = raw_incident_energy[indices]

    incident_energy = _flatten_incident_energy(incident_energy)

    events = np.transpose(events, (0, 2, 1))

    # Create mask for valid hits (right-aligned)
    max_hits = events.shape[1]
    hit_indices = np.arange(max_hits)
    # Valid if index >= max_hits - n_points
    # padding is at the beginning
    start_indices = max_hits - n_points
    mask = hit_indices[None, :] >= start_indices[:, None]

    # Flatten
    events_flat = events.reshape(-1, 5)
    mask_flat = mask.flatten()

    valid_events = events_flat[mask_flat]

    # Extract features
    flat_x = valid_events[:, 0]
    flat_z = valid_events[:, 1]
    flat_y = valid_events[:, 2]
    flat_energy = valid_events[:, 3]

    counts = n_points.astype(int)

    ak_x = ak.unflatten(flat_x, counts)
    ak_y = ak.unflatten(flat_y, counts)
    ak_z = ak.unflatten(flat_z, counts)
    ak_energy = ak.unflatten(flat_energy, counts)

    ak_array = ak.zip({"x": ak_x, "y": ak_y, "z": ak_z, "energy": ak_energy}, depth_limit=1)

    return ak_array, incident_energy


def _load_separated_showers(h5file, n_load, random_seed, start=None, stop=None, step=None):
    ds_energy = cast(h5py.Dataset, h5file["energy"])  # type: ignore[index]
    ds_x = cast(h5py.Dataset, h5file["x"])  # type: ignore[index]
    ds_y = cast(h5py.Dataset, h5file["y"])  # type: ignore[index]
    ds_z = cast(h5py.Dataset, h5file["z"])  # type: ignore[index]

    if "incident_energy" in h5file:
        ds_incident_energy = cast(h5py.Dataset, h5file["incident_energy"])  # type: ignore[index]
    elif "genE" in h5file:
        ds_incident_energy = cast(h5py.Dataset, h5file["genE"])  # type: ignore[index]
    else:
        raise KeyError("Incident energy not found.")

    total_showers = int(ds_energy.shape[0])
    row_selector = _make_row_selector(
        total_showers, n_load=n_load, start=start, stop=stop, step=step, random_seed=random_seed
    )

    ds_energy = np.asarray(ds_energy[row_selector])
    ds_x = np.asarray(ds_x[row_selector])
    ds_y = np.asarray(ds_y[row_selector])
    ds_z = np.asarray(ds_z[row_selector])
    raw_incident_energy = np.asarray(ds_incident_energy[row_selector])

    incident_energy = _flatten_incident_energy(raw_incident_energy)

    mask = ds_energy > 0
    counts = np.sum(mask, axis=1)

    flat_energy = ds_energy[mask]
    flat_x = ds_x[mask]
    flat_y = ds_y[mask]
    flat_z = ds_z[mask]

    ak_energy = ak.unflatten(flat_energy, counts)
    ak_x = ak.unflatten(flat_x, counts)
    ak_y = ak.unflatten(flat_y, counts)
    ak_z = ak.unflatten(flat_z, counts)

    ak_array = ak.zip({"x": ak_x, "y": ak_y, "z": ak_z, "energy": ak_energy}, depth_limit=1)

    return ak_array, incident_energy


def _flatten_incident_energy(raw_energy: np.ndarray) -> np.ndarray:
    energy = np.asarray(raw_energy)
    if energy.ndim == 1:
        return energy
    if energy.ndim == 2:
        return energy.reshape(energy.shape[0], -1)[:, 0]
    raise ValueError(f"Unexpected incident energy tensor shape: {energy.shape}")


def _load_cartesian_showers(
    dataset_showers: np.ndarray,
    incident_energy: np.ndarray,
    n_load: int,
    chunk_size: int,
) -> tuple[ak.Array, np.ndarray]:
    if n_load is None:
        n_load = int(dataset_showers.shape[0])
    data_dict = {"x": [], "y": [], "z": [], "energy": []}
    energy_chunks: list[np.ndarray] = []

    for start in range(0, n_load, chunk_size):
        end = min(start + chunk_size, n_load)
        table = np.array(dataset_showers[start:end], copy=True)

        data_dict["x"].append(table[:, :, 0])
        data_dict["y"].append(table[:, :, 1])
        data_dict["z"].append(table[:, :, 2])
        data_dict["energy"].append(table[:, :, 3])
        energy_chunks.append(incident_energy[start:end])

    concatenated = {key: ak.concatenate(value) for key, value in data_dict.items()}
    ak_array = ak.Array(concatenated)
    energy_array = np.concatenate(energy_chunks)
    return ak_array, energy_array


def standardize_shower_schema(ak_showers: ak.Array, dataset_type: str) -> ak.Array:
    try:
        standardize = GEOMETRY_STANDARDIZERS[dataset_type]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset_type: {dataset_type}") from exc
    return standardize(ak_showers)
