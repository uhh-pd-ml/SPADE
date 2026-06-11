"""Generate calorimeter showers from a trained humite checkpoint.

This mirrors the generation path used during training (the generative-evaluation
callback): it rebuilds the model from the Hydra config, loads the checkpoint
weights, samples incident energies, and autoregressively generates showers with
the stop head deciding each shower's length. Decoded physical hits are written to
an HDF5 file.

The architecture is rebuilt from the ``config.yaml`` bundled next to the
checkpoint, so each checkpoint just works without remembering its overrides.

Examples
--------
    # generate 1000 showers with energies uniform in [10, 100] GeV
    python -m humite.cli.generate \
        +generate.checkpoint=checkpoints/spade/getting_high/model.ckpt

    # fixed incident energy, custom count/output
    python -m humite.cli.generate \
        +generate.checkpoint=checkpoints/spade/getting_high/model.ckpt \
        +generate.energy_gev=50 +generate.n_showers=5000 \
        +generate.output=showers_50gev.h5

``config.yaml`` next to the checkpoint is loaded automatically (it replaces the
``model``, ``dataset`` and sampling settings with the ones the checkpoint was
trained with). Point elsewhere with ``+generate.config=/path/to/config.yaml``.
"""

from __future__ import annotations

# Load a repo-root .env (HUMITE_DATA_DIR, HUMITE_OUTPUT_DIR, COMET_API_TOKEN, ...)
# before Hydra resolves ${oc.env:...} interpolations. usecwd=True searches up
# from the directory the command was launched in.
try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ModuleNotFoundError:
    pass

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from humite.core.config.schemas import DatasetConfig
from humite.core.data.preprocessing.adapters import build_adapter
from humite.core.data.preprocessing.utils import apply_feature_transform
from humite.core.utils.environment_setup import initialize_environment
from humite.core.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

# Sampling keys that must not be forwarded to ``generate_physical_showers`` as
# extra kwargs: ``batch_size`` collides with the positional argument, and the
# diagnostic switch forces exact hit counts (not wanted for real generation).
_SAMPLING_DROP_KEYS = ("batch_size", "force_n_hits_for_diagnostic")


def _load_checkpoint_weights(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load a Lightning/torch checkpoint into ``model`` (non-strict).

    Checkpoints saved during training already contain the weights used for the
    monitored ``val/loss`` (EMA weights when the EMA callback is active), so the
    state dict can be loaded directly.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # nosec - trusted ckpt
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning(
            "Missing keys when loading checkpoint (%d), e.g. %s", len(missing), missing[:5]
        )
    if unexpected:
        log.warning(
            "Unexpected keys when loading checkpoint (%d), e.g. %s",
            len(unexpected),
            unexpected[:5],
        )
    log.info("Loaded checkpoint weights from %s", ckpt_path)


def _sampling_kwargs(cfg: DictConfig) -> Dict[str, Any]:
    """Resolve the generation sampling settings from the training config."""
    sampling_node = OmegaConf.select(cfg, "trainer.generative_eval.sampling")
    sampling: Dict[str, Any] = {}
    if sampling_node is not None:
        sampling = dict(OmegaConf.to_container(sampling_node, resolve=True))  # type: ignore[arg-type]
    for key in _SAMPLING_DROP_KEYS:
        sampling.pop(key, None)
    return sampling


def _incident_energies_gev(gcfg: Dict[str, Any], n_showers: int, seed: int) -> np.ndarray:
    """Sample incident energies in GeV (fixed value or uniform range)."""
    fixed = gcfg.get("energy_gev", None)
    if fixed is not None:
        return np.full(n_showers, float(fixed), dtype=np.float32)
    e_min = float(gcfg.get("energy_min_gev", 10.0))
    e_max = float(gcfg.get("energy_max_gev", 100.0))
    rng = np.random.default_rng(seed)
    return rng.uniform(e_min, e_max, size=n_showers).astype(np.float32)


def _build_model_and_adapter(
    cfg: DictConfig, ckpt_path: str, device: torch.device
) -> tuple[torch.nn.Module, Any, Dict[str, Any]]:
    """Rebuild the model from ``cfg`` (model + dataset), load the checkpoint, and
    build the adapter that decodes generated tokens back to physical (x, y, z, E)."""
    dataset_spec = instantiate(cfg.dataset)
    if hasattr(cfg.dataset, "features") and not getattr(dataset_spec, "features", None):
        dataset_spec.features = OmegaConf.to_container(cfg.dataset.features, resolve=True)
    model = instantiate(cfg.model, spec=dataset_spec)
    _load_checkpoint_weights(model, ckpt_path)
    model.to(device).eval()

    dataset_dict = dict(OmegaConf.to_container(cfg.dataset, resolve=True))  # type: ignore[arg-type]
    dataset_dict.pop("_target_", None)
    adapter = build_adapter(DatasetConfig(**dataset_dict))
    return model, adapter, dataset_dict


def _run_generation(
    model: torch.nn.Module,
    adapter: Any,
    x_shower_all: torch.Tensor,
    incident_cond_all: torch.Tensor,
    n_showers: int,
    batch_size: int,
    sampling: Dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Autoregressively generate ``n_showers`` showers in batches and return the
    decoded ``hit_features`` (n, max_hits, 4) and ``hit_mask`` (n, max_hits).

    ``x_shower_all`` is the incident energy in **raw GeV** (used by the physics-stop
    head, which divides the cumulative energy by it); ``incident_cond_all`` is the
    **transformed** incident energy used for model conditioning. Passing the
    transformed value as ``x_shower`` makes the physics-stop ratio explode and
    collapses shower length, so the two must be kept distinct."""
    hit_features_chunks: list[np.ndarray] = []
    hit_mask_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n_showers, batch_size):
            end = min(start + batch_size, n_showers)
            x_shower = x_shower_all[start:end].to(device)
            batch_sampling = dict(sampling)
            batch_sampling["incident_energy_cond"] = incident_cond_all[start:end].to(device)
            out = model.generate_physical_showers(
                batch_size=end - start,
                adapter=adapter,
                x_shower=x_shower,
                **batch_sampling,
            )
            if "hit_features" not in out:
                raise RuntimeError(
                    "Generation output has no 'hit_features'; an adapter is required to decode "
                    "showers to physical space."
                )
            hit_features_chunks.append(out["hit_features"].detach().cpu().numpy())
            if "hit_mask" in out:
                hit_mask_chunks.append(out["hit_mask"].detach().cpu().numpy())
            log.info("  generated %d / %d", end, n_showers)

    hit_features = _concat_padded(hit_features_chunks)
    hit_mask = _concat_padded(hit_mask_chunks) if hit_mask_chunks else None
    return hit_features, hit_mask


# Energy-sum lower-envelope filter (paper App. "Postprocessing"): reject and
# regenerate showers whose total deposited energy E_sum falls below
# s*(a*E_inc^b + c). Keys mirror scripts/plotting/model_runner.py for config
# compatibility. Coefficients must be in the SAME energy units as the decoded
# hits (the sum of hit_features[..., 3]).
_ENERGY_SUM_FILTER_KEYS = (
    "enable_energy_sum_filter",
    "energy_sum_power_a",
    "energy_sum_power_b",
    "energy_sum_power_c",
    "energy_sum_safety_factor",
    "energy_sum_max_retries",
)


def _pop_energy_sum_filter(sampling: Dict[str, Any]) -> Dict[str, Any]:
    """Remove the energy-sum-filter keys from ``sampling`` (they must not be
    forwarded to ``generate_physical_showers``) and return them."""
    return {k: sampling.pop(k) for k in _ENERGY_SUM_FILTER_KEYS if k in sampling}


def _energy_sum_per_shower(hit_features: np.ndarray, hit_mask: Optional[np.ndarray]) -> np.ndarray:
    e = hit_features[..., 3]
    if hit_mask is not None:
        e = np.where(hit_mask.astype(bool), e, 0.0)
    return e.sum(axis=1)


def _pad_hits(arr: np.ndarray, target_len: int) -> np.ndarray:
    if arr.shape[1] >= target_len:
        return arr
    pad = [(0, 0), (0, target_len - arr.shape[1])] + [(0, 0)] * (arr.ndim - 2)
    return np.pad(arr, pad)


def _apply_energy_sum_filter(
    model: torch.nn.Module,
    adapter: Any,
    hit_features: np.ndarray,
    hit_mask: Optional[np.ndarray],
    e_gev: np.ndarray,
    incident_cond_all: torch.Tensor,
    sampling: Dict[str, Any],
    fcfg: Dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Flag showers below the Geant4 lower envelope and regenerate them in place,
    up to ``energy_sum_max_retries`` rounds (paper App. Postprocessing)."""
    a = float(fcfg.get("energy_sum_power_a", 0.0))
    b = float(fcfg.get("energy_sum_power_b", 1.0))
    c = float(fcfg.get("energy_sum_power_c", 0.0))
    s = float(fcfg.get("energy_sum_safety_factor", 0.9))
    k_max = int(fcfg.get("energy_sum_max_retries", 2))
    inc = np.clip(e_gev.astype(np.float64), 1e-3, None)
    threshold = s * (a * inc**b + c)

    n0 = int((_energy_sum_per_shower(hit_features, hit_mask) < threshold).sum())
    log.info(
        "energy-sum filter active (a=%.4g, b=%.4g, c=%.4g, s=%.2f): %d/%d showers below envelope",
        a,
        b,
        c,
        s,
        n0,
        len(e_gev),
    )
    for attempt in range(k_max):
        flagged = _energy_sum_per_shower(hit_features, hit_mask) < threshold
        n = int(flagged.sum())
        if n == 0:
            break
        log.info(
            "energy-sum filter: %d/%d showers below envelope; regenerating (round %d/%d)",
            n,
            len(e_gev),
            attempt + 1,
            k_max,
        )
        idx = np.where(flagged)[0]
        x_raw = torch.from_numpy(e_gev[idx]).float().view(-1, 1)
        new_feats, new_mask = _run_generation(
            model, adapter, x_raw, incident_cond_all[idx], len(idx), batch_size, sampling, device
        )
        target_len = max(hit_features.shape[1], new_feats.shape[1])
        hit_features = _pad_hits(hit_features, target_len)
        new_feats = _pad_hits(new_feats, target_len)
        hit_features[idx] = new_feats
        if hit_mask is not None and new_mask is not None:
            hit_mask = _pad_hits(hit_mask, target_len)
            new_mask = _pad_hits(new_mask, target_len)
            hit_mask[idx] = new_mask

    residual = int((_energy_sum_per_shower(hit_features, hit_mask) < threshold).sum())
    if residual:
        log.warning(
            "energy-sum filter: %d showers still below envelope after %d rounds; kept as-is",
            residual,
            k_max,
        )
    return hit_features, hit_mask


def generate_showers(
    checkpoint: str,
    config: Optional[str] = None,
    *,
    n_showers: int = 1000,
    batch_size: int = 512,
    energy_gev: Optional[float] = None,
    energy_min_gev: float = 10.0,
    energy_max_gev: float = 100.0,
    seed: int = 42,
    device: Optional[str] = None,
    energy_sum_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """Generate showers programmatically (e.g. from a notebook).

    Rebuilds the model from the checkpoint's bundled ``config`` (the minimal
    ``model``/``dataset``/``sampling`` config shipped next to each checkpoint),
    loads the weights, samples incident energies, and autoregressively generates
    showers. If ``config`` is omitted it defaults to ``config.yaml`` sitting next
    to the checkpoint, so usually only ``checkpoint`` is needed.

    Returns a dict with ``hit_features`` (n, max_hits, 4=[x, y, z, E]),
    ``hit_mask`` (n, max_hits) flagging real hits, and ``incident_energy`` (n,)
    in GeV. Pass a fixed ``energy_gev`` or a ``[energy_min_gev, energy_max_gev]``
    range.
    """
    # Resolvers used inside the config (e.g. ${eval:}) + reproducible sampling.
    from humite.core.utils.environment_setup import register_omegaconf_resolvers

    register_omegaconf_resolvers()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg_path = Path(config) if config is not None else Path(checkpoint).with_name("config.yaml")
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"No config at {cfg_path}. Pass config=... or keep config.yaml next to the checkpoint."
        )
    cfg = OmegaConf.load(str(cfg_path))
    dev = (
        torch.device(device)
        if device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, adapter, dataset_dict = _build_model_and_adapter(cfg, str(checkpoint), dev)

    gcfg = {
        "energy_gev": energy_gev,
        "energy_min_gev": energy_min_gev,
        "energy_max_gev": energy_max_gev,
    }
    e_gev = _incident_energies_gev(gcfg, n_showers, seed)
    inc_params = (dataset_dict.get("features") or {}).get("incident_energy", {}) or {}
    # x_shower = raw GeV (physics-stop divides cumulative energy by it);
    # incident_energy_cond = transformed value the model was conditioned on.
    x_shower_all = torch.from_numpy(e_gev).float().view(-1, 1)
    incident_cond_all = (
        apply_feature_transform(torch.from_numpy(e_gev), inc_params).float().view(-1, 1)
    )

    sampling = _sampling_kwargs(cfg)
    # Energy-sum filter config: from the bundled config's sampling block, with an
    # optional explicit override taking precedence.
    fcfg = _pop_energy_sum_filter(sampling)
    if energy_sum_filter:
        fcfg.update(energy_sum_filter)
    hit_features, hit_mask = _run_generation(
        model, adapter, x_shower_all, incident_cond_all, n_showers, batch_size, sampling, dev
    )
    if fcfg.get("enable_energy_sum_filter"):
        hit_features, hit_mask = _apply_energy_sum_filter(
            model,
            adapter,
            hit_features,
            hit_mask,
            e_gev,
            incident_cond_all,
            sampling,
            fcfg,
            batch_size,
            dev,
        )
    out: Dict[str, np.ndarray] = {"hit_features": hit_features, "incident_energy": e_gev}
    if hit_mask is not None:
        out["hit_mask"] = hit_mask
    return out


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.getLogger("comet_ml").setLevel(logging.WARNING)
    initialize_environment(cfg)  # registers OmegaConf resolvers used in the config

    gcfg: Dict[str, Any] = {}
    if OmegaConf.select(cfg, "generate") is not None:
        gcfg = dict(OmegaConf.to_container(cfg.generate, resolve=True))  # type: ignore[arg-type]

    ckpt_path = gcfg.get("checkpoint")
    if not ckpt_path:
        raise ValueError(
            "No checkpoint given. Pass it with '+generate.checkpoint=/path/to/model.ckpt'."
        )

    # Load the checkpoint's bundled config so the rebuilt architecture matches the
    # weights. Defaults to config.yaml sitting next to the checkpoint; this replaces
    # the model / dataset / sampling nodes with the ones it was trained with.
    gen_config_path = gcfg.get("config")
    if not gen_config_path:
        sibling = Path(str(ckpt_path)).with_name("config.yaml")
        if sibling.is_file():
            gen_config_path = str(sibling)
            log.info("Using config bundled next to checkpoint: %s", gen_config_path)
    if gen_config_path:
        if not Path(str(gen_config_path)).is_file():
            raise FileNotFoundError(
                f"+generate.config points at a missing file: {gen_config_path}"
            )
        loaded = OmegaConf.load(str(gen_config_path))
        OmegaConf.set_struct(cfg, False)
        for node in ("model", "dataset"):
            node_cfg = OmegaConf.select(loaded, node)
            if node_cfg is not None:
                cfg[node] = node_cfg
        loaded_sampling = OmegaConf.select(loaded, "trainer.generative_eval.sampling")
        if loaded_sampling is not None:
            if OmegaConf.select(cfg, "trainer") is None:
                cfg.trainer = {}
            if OmegaConf.select(cfg, "trainer.generative_eval") is None:
                cfg.trainer.generative_eval = {}
            cfg.trainer.generative_eval.sampling = loaded_sampling
        OmegaConf.set_struct(cfg, True)
        log.info("Loaded model/dataset/sampling from bundled config %s", gen_config_path)

    n_showers = int(gcfg.get("n_showers", 1000))
    batch_size = int(gcfg.get("batch_size", 512))
    out_path = Path(str(gcfg.get("output", "generated_showers.h5")))
    seed = int(OmegaConf.select(cfg, "seed") or 42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1-2. Rebuild the model from config + load the checkpoint, and build the
    #      adapter that decodes generated tokens back to physical x, y, z, E.
    model, adapter, dataset_dict = _build_model_and_adapter(cfg, str(ckpt_path), device)

    # 3. Incident energies: sample in GeV. x_shower stays in raw GeV (the
    #    physics-stop head divides the cumulative energy by it); the transformed
    #    value is passed separately as incident_energy_cond for model conditioning.
    e_gev = _incident_energies_gev(gcfg, n_showers, seed)
    inc_params = (dataset_dict.get("features") or {}).get("incident_energy", {}) or {}
    x_shower_all = torch.from_numpy(e_gev).float().view(-1, 1)
    incident_cond_all = (
        apply_feature_transform(torch.from_numpy(e_gev), inc_params).float().view(-1, 1)
    )

    sampling = _sampling_kwargs(cfg)
    # Energy-sum filter: from the bundled sampling block, overridable via +generate.*
    fcfg = _pop_energy_sum_filter(sampling)
    for key in _ENERGY_SUM_FILTER_KEYS:
        if key in gcfg:
            fcfg[key] = gcfg[key]
    log.info(
        "Generating %d showers (batch_size=%d) on %s; sampling=%s",
        n_showers,
        batch_size,
        device,
        sampling,
    )

    # 4. Generate in batches.
    hit_features, hit_mask = _run_generation(
        model, adapter, x_shower_all, incident_cond_all, n_showers, batch_size, sampling, device
    )

    # 4b. Optional energy-sum lower-envelope filter (paper App. Postprocessing).
    if fcfg.get("enable_energy_sum_filter"):
        hit_features, hit_mask = _apply_energy_sum_filter(
            model,
            adapter,
            hit_features,
            hit_mask,
            e_gev,
            incident_cond_all,
            sampling,
            fcfg,
            batch_size,
            device,
        )

    # 5. Save. hit_features is (n_showers, max_hits, 4=[x, y, z, E]); hit_mask flags real hits.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("hit_features", data=hit_features, compression="gzip")
        if hit_mask is not None:
            f.create_dataset("hit_mask", data=hit_mask.astype(np.bool_), compression="gzip")
        f.create_dataset("incident_energy", data=e_gev)
        f.attrs["n_showers"] = n_showers
        f.attrs["checkpoint"] = str(ckpt_path)
        f.attrs["feature_order"] = "x,y,z,energy"
    log.info("Wrote %d showers to %s", n_showers, out_path)


def _concat_padded(chunks: list[np.ndarray]) -> np.ndarray:
    """Concatenate per-batch arrays, right-padding the hit axis to the global max."""
    if not chunks:
        return np.empty((0,))
    if len(chunks) == 1:
        return chunks[0]
    max_hits = max(c.shape[1] for c in chunks)
    padded = []
    for c in chunks:
        if c.shape[1] < max_hits:
            pad = [(0, 0), (0, max_hits - c.shape[1])] + [(0, 0)] * (c.ndim - 2)
            c = np.pad(c, pad)
        padded.append(c)
    return np.concatenate(padded, axis=0)


if __name__ == "__main__":
    main()  # type: ignore[misc]
