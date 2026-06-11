import math
from typing import Any, Dict, Optional, Tuple

import awkward as ak
import numpy as np
import torch
import vector

# humite/core/utils/logger.py
from ...utils.pylogger import get_pylogger

vector.register_awkward()
pylogger = get_pylogger(__name__)


def ak_pad(x: ak.Array, maxlen: int, axis: int = 1, fill_value=0, return_mask=False):
    """Function to pad an awkward array to a specified length. The array is padded along the
    specified axis.

    Parameters
    ----------
    x : awkward array
        Array to pad.
    maxlen : int
        Length to pad to.
    axis : int, optional
        Axis along which to pad. Default is 1.
    fill_value : float or int, optional
        Value to use for padding. Default is 0.
    return_mask : bool, optional
        If True, also return a mask array indicating which values are padded.
        Default is False.
        If the input array has fields, the mask is created from the first field.

    Returns
    -------
    awkward array
        Padded array.
    mask : awkward array
        Mask array indicating which values are padded. Only returned if return_mask is True.
    """
    padded_x = ak.fill_none(ak.pad_none(x, maxlen, axis=axis, clip=True), fill_value)
    if return_mask:
        if len(x.fields) >= 1:
            mask = ak.ones_like(x[x.fields[0]], dtype="bool")
        else:
            mask = ak.ones_like(x, dtype="bool")
        mask = ak.fill_none(ak.pad_none(mask, maxlen, axis=axis, clip=True), False)
        return padded_x, mask
    return padded_x


def get_valid_energy_mask(x: ak.Array, energy_threshold: float) -> ak.Array:
    """Creates a boolean mask based on energy threshold.

    Args:
        x (ak.Array): Input array with "energy" field
        energy_threshold (float): Threshold value

    Returns:
        ak.Array: Boolean mask
    """
    energy = x["energy"]
    return (energy > energy_threshold) & np.isfinite(energy)


def ak_padding(x: ak.Array, maxlen: int, energy_threshold: float):
    """Pads an Awkward Array and creates a mask based on energy values.

    Args:
        x (ak.Array): The Awkward Array to pad.
        maxlen (int): The maximum length to pad to.
        energy_threshold (float): The threshold for considering energy as non-zero.

    Returns:
        tuple: A tuple containing the padded array and the mask.
    """

    # Create mask based on energy threshold
    mask = get_valid_energy_mask(x, energy_threshold)

    # Pad and mask to maxlen
    padded_mask = ak.pad_none(mask, maxlen, axis=1, clip=True)
    return padded_mask


def ak_to_np_stack(ak_array: ak.Array, names: Optional[list] = None, axis: int = -1):
    """Function to convert an awkward array to a numpy array by stacking the values of the
    specified fields. This is much faster than ak.to_numpy(ak_array) for large arrays.

    Parameters
    ----------
    ak_array : awkward array
        Array to convert.
    names : list, optional
        List of field names to convert. Default is None.
    axis : int, optional
        Axis along which to stack the values. Default is -1.
    """
    if names is None:
        raise ValueError("names must be specified")
    return ak.to_numpy(
        np.stack(
            [ak.to_numpy(ak.values_astype(ak_array[name], "float32")) for name in names],
            axis=axis,
        )
    )


def _is_awkward(x) -> bool:
    return ak is not None and isinstance(x, ak.highlevel.Array)


def _apply_op(op, x, backend: str):
    """
    op: callable or string ("np.log", "torch.exp", "log", etc.)
    backend: "torch" or "numpy"
    """
    if op is None:
        return x
    if callable(op):
        return op(x)
    if not isinstance(op, str):
        raise TypeError(f"func/inv_func must be a callable or str, got {type(op)}")

    name = op
    if name.startswith("np."):
        name = name[3:]
    elif name.startswith("torch."):
        name = name[6:]

    if backend == "torch":
        if not hasattr(torch, name):
            raise ValueError(f"torch has no op '{name}' (from '{op}')")
        return getattr(torch, name)(x)

    # numpy backend (works for np.ndarray and ak.Array because of ufunc dispatch)
    if not hasattr(np, name):
        raise ValueError(f"numpy has no op '{name}' (from '{op}')")
    return getattr(np, name)(x)


def apply_feature_transform(data, params, invert: bool = False):
    """
    Supports:
      - torch.Tensor (stays torch, stays on device)
      - numpy.ndarray
      - awkward.highlevel.Array (uses numpy ufuncs; returns awkward)

    Forward:  func -> affine
    Inverse:  inverse affine -> inv_func
    """
    subtract_by = params.get("subtract_by", 0.0)
    multiply_by = params.get("multiply_by", 1.0)
    func = params.get("func", None)
    inv_func = params.get("inv_func", None)

    # 1) First: determine type/backend
    if isinstance(data, torch.Tensor):
        backend = "torch"
        kind = "torch"
    elif isinstance(data, np.ndarray):
        backend = "numpy"
        kind = "numpy"
    elif _is_awkward(data):
        backend = "numpy"
        kind = "awkward"
    else:
        raise TypeError(
            f"data must be torch.Tensor, np.ndarray, or awkward.Array; got {type(data)}"
        )

    # 2) Apply pre/postprocessing
    if not invert:
        data = _apply_op(func, data, backend)
        data = (data - subtract_by) * multiply_by
    else:
        data = data / multiply_by + subtract_by
        data = _apply_op(inv_func, data, backend)

    return data


def build_hcal_resolution_bin_edges(
    e_min: float,
    e_max: float,
    A: float = 0.50,
    B: float = 0.0,
    C: float = 0.02,
    step_sigma: float = 0.5,
) -> np.ndarray:
    """HCal resolution-based bin edges on raw GeV.

    sigma_E(E) = sqrt(A^2 * E + B^2 + C^2 * E^2)
    Greedy walk from e_min:  E_{k+1} = E_k + step_sigma * sigma_E(E_k)
    Guarantees the last edge >= e_max.
    """
    if not (e_max > e_min > 0.0):
        raise ValueError(f"require 0 < e_min < e_max, got e_min={e_min}, e_max={e_max}")
    if step_sigma <= 0.0:
        raise ValueError(f"step_sigma must be positive, got {step_sigma}")
    a2, b2, c2 = float(A) ** 2, float(B) ** 2, float(C) ** 2

    edges: list[float] = [float(e_min)]
    E = float(e_min)
    max_iter = 10_000_000
    for _ in range(max_iter):
        sigma = math.sqrt(max(a2 * E + b2 + c2 * E * E, 1e-18))
        step = step_sigma * sigma
        if step <= 0.0:
            raise RuntimeError("bin step became non-positive; check A/B/C")
        E = E + step
        edges.append(E)
        if E >= e_max:
            break
    else:
        raise RuntimeError("hcal bin generation did not reach e_max within iteration cap")
    return np.asarray(edges, dtype=np.float64)


def bin_centers_from_edges(edges: np.ndarray) -> np.ndarray:
    edges_arr = np.asarray(edges, dtype=np.float64)
    if edges_arr.ndim != 1 or edges_arr.size < 2:
        raise ValueError("edges must be 1-D with length >= 2")
    return 0.5 * (edges_arr[:-1] + edges_arr[1:])


def summarize_value(v: Any) -> Any:
    if isinstance(v, str):
        return v
    if hasattr(v, "shape"):
        try:
            return tuple(v.shape)  # type: ignore[attr-defined]
        except Exception:
            return type(v).__name__
    if isinstance(v, (bool, int, float)):
        return v
    if hasattr(v, "__len__"):
        try:
            return len(v)  # type: ignore[arg-type]
        except Exception:
            return type(v).__name__
    return type(v).__name__


def decompose_combined_spatial_tokens(
    combined_tokens: torch.Tensor,
    axis_nbins: Dict[str, int],
    axis_names: Tuple[str, ...],
    missing_id_spatial: Optional[int] = None,
    spatial_vocab: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    if combined_tokens.dtype != torch.long:
        combined_tokens = combined_tokens.to(torch.long)

    has_missing_id = missing_id_spatial is not None
    has_spatial_vocab = spatial_vocab is not None
    if has_missing_id != has_spatial_vocab:
        raise ValueError("missing_id_spatial and spatial_vocab must be provided together")

    missing_mask = combined_tokens.eq(0)
    if missing_id_spatial is not None and spatial_vocab is not None:
        missing_mask = missing_mask | combined_tokens.eq(int(missing_id_spatial))
        missing_mask = missing_mask | combined_tokens.ge(int(spatial_vocab))

    valid_tokens = torch.where(missing_mask, torch.ones_like(combined_tokens), combined_tokens)
    running_values = valid_tokens - 1
    axis_tokens: Dict[str, torch.Tensor] = {}
    for axis_name in axis_names:
        nbins = int(axis_nbins[axis_name])
        axis_value = running_values % nbins
        axis_tokens[axis_name] = axis_value + 1
        running_values = torch.div(running_values, nbins, rounding_mode="floor")
    return axis_tokens, missing_mask
