from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

from ..data.datasets.calorimeter_iterable import CalorimeterIterableDataset


def build_dataset(cfg: Dict[str, Any] | DictConfig | Any, spec: Optional[Any] = None):
    name: Optional[str]
    kwargs: Dict[str, Any]
    preproc: Any

    if isinstance(cfg, DictConfig):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]
        assert isinstance(cfg_dict, dict)
        name = cfg_dict.get("name")
        preproc = cfg_dict.get("preprocessor")
        kwargs = {k: v for k, v in cfg_dict.items() if k not in {"name", "preprocessor"}}
    elif isinstance(cfg, dict):
        name = cfg.get("name")
        preproc = cfg.get("preprocessor")
        kwargs = {k: v for k, v in cfg.items() if k not in {"name", "preprocessor"}}
    else:
        name = getattr(cfg, "name", None)
        preproc = getattr(cfg, "preprocessor", None)
        kwargs = getattr(cfg, "kwargs", {}) or {
            k: getattr(cfg, k)
            for k in dir(cfg)
            if not k.startswith("_") and k not in {"name", "preprocessor"}
        }

    if not name:
        raise KeyError("Dataset config must include a 'name'.")

    # Build calorimeter iterable datasets container when requested
    if name in {"calo_iterable", "calorimeter_iterable"}:

        def to_namespace(obj: Any) -> Any:
            if isinstance(obj, dict):
                return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
            return obj

        cfg_data_dict = kwargs | {"preprocessor": preproc} if preproc is not None else kwargs
        # If an ExperimentSpec is provided, expose a few convenience fields expected by downstream code
        if spec is not None:
            try:
                geom = getattr(spec, "geometry", None)
                if geom is not None and "nbins_x" not in cfg_data_dict:
                    nbx, nby, nbz = geom.nbins  # type: ignore[assignment]
                    cfg_data_dict.setdefault("nbins_x", int(nbx))
                    cfg_data_dict.setdefault("nbins_y", int(nby))
                    cfg_data_dict.setdefault("nbins_z", int(nbz))
                cfg_data_dict.setdefault(
                    "use_energy_tokenization",
                    bool(getattr(spec, "use_energy_tokenization", False)),
                )
            except Exception:
                pass

        cfg_ns = SimpleNamespace(data=to_namespace(cfg_data_dict))

        train = CalorimeterIterableDataset(cfg_ns.data.train_files, cfg_ns, repeat=True)  # type: ignore[arg-type]
        val = CalorimeterIterableDataset(cfg_ns.data.val_files, cfg_ns, repeat=False)  # type: ignore[arg-type]
        test = CalorimeterIterableDataset(cfg_ns.data.test_files, cfg_ns, repeat=False)  # type: ignore[arg-type]
        return SimpleNamespace(train=train, val=val, test=test)

    raise KeyError(
        f"Unknown dataset '{name}'. Available: ['calo_iterable', 'calorimeter_iterable']"
    )
