from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import lightning as L
from torch.utils.data import DataLoader, get_worker_info

from humite.core.config.schemas import DatasetConfig
from humite.core.data.datasets.calorimeter_iterable import CalorimeterIterableDataset
from humite.core.data.preprocessing.adapters import build_adapter
from humite.core.data.preprocessing.collators import BatchCollator
from humite.core.data.preprocessing.utils import summarize_value
from humite.core.utils.pylogger import get_pylogger


class CaloDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: DatasetConfig,
        batch_size: int = 1,
        num_workers: int = 5,
        val_num_workers: int | None = None,
        test_num_workers: int | None = None,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        val_pin_memory: bool | None = None,
        test_pin_memory: bool | None = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.collator = BatchCollator(dataset)
        self.adapter = build_adapter(dataset)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.val_num_workers = (
            int(val_num_workers) if val_num_workers is not None else self.num_workers
        )
        self.test_num_workers = (
            int(test_num_workers) if test_num_workers is not None else self.num_workers
        )
        self.prefetch_factor = int(prefetch_factor)
        self.pin_memory = bool(pin_memory)
        self.val_pin_memory = (
            bool(val_pin_memory) if val_pin_memory is not None else self.pin_memory
        )
        self.test_pin_memory = (
            bool(test_pin_memory) if test_pin_memory is not None else self.pin_memory
        )
        self.log = get_pylogger(__name__)
        self.has_logged_initial_batch_overview = False

    def _collate_and_adapt(self, batch_list):
        batch = self.collator(batch_list)
        batch = self.adapter(batch)

        if not self.has_logged_initial_batch_overview:
            worker_info = get_worker_info()
            if worker_info is None or worker_info.id == 0:
                self.log.debug(f"Collated batch keys: {list(batch.keys())}")
                batch_value_summary = {k: summarize_value(v) for k, v in batch.items()}
                self.log.debug(f"Batch value summary: {batch_value_summary}")
            self.has_logged_initial_batch_overview = True

        return batch

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = CalorimeterIterableDataset(
            self.dataset.train_files, self.dataset, repeat=True
        )
        self.val_ds = CalorimeterIterableDataset(
            self.dataset.val_files, self.dataset, repeat=False
        )
        self.test_ds = CalorimeterIterableDataset(
            self.dataset.test_files, self.dataset, repeat=False
        )

    def train_dataloader(self) -> DataLoader:
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "collate_fn": self._collate_and_adapt,
            "pin_memory": self.pin_memory,
            "persistent_workers": bool(self.num_workers > 0),
        }
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return DataLoader(self.train_ds, **kwargs)

    def val_dataloader(self) -> DataLoader:
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "num_workers": self.val_num_workers,
            "collate_fn": self._collate_and_adapt,
            "pin_memory": self.val_pin_memory,
            "persistent_workers": bool(self.val_num_workers > 0),
        }
        if self.val_num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return DataLoader(self.val_ds, **kwargs)

    def test_dataloader(self) -> DataLoader:
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "num_workers": self.test_num_workers,
            "collate_fn": self._collate_and_adapt,
            "pin_memory": self.test_pin_memory,
            "persistent_workers": bool(self.test_num_workers > 0),
        }
        if self.test_num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return DataLoader(self.test_ds, **kwargs)


@dataclass
class DataModuleCfg:
    batch_size: int = 1
    num_workers: int = 0
    val_num_workers: int | None = None
    test_num_workers: int | None = None
    prefetch_factor: int = 2
    pin_memory: bool = True
    val_pin_memory: bool | None = None
    test_pin_memory: bool | None = None


def _to_plain_dict(obj: Any) -> dict:
    try:
        # OmegaConf DictConfig
        from omegaconf import OmegaConf  # type: ignore

        if hasattr(obj, "_content") or str(type(obj)).endswith("DictConfig'>"):
            try:
                return OmegaConf.to_container(obj, resolve=True)  # type: ignore[arg-type]
            except Exception:
                pass
    except Exception:
        pass

    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return obj.model_dump()  # pydantic v2
        except Exception:
            pass
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return obj.dict()  # pydantic v1
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        except Exception:
            pass
    return {}


def build_dm_cfg(cfg: Any) -> DataModuleCfg:
    def section(name: str) -> dict:
        sec = getattr(cfg, name, None)
        if sec is None:
            return {}
        return _to_plain_dict(sec)

    merged = {**section("trainer"), **section("training")}
    allowed = {
        "batch_size",
        "num_workers",
        "val_num_workers",
        "test_num_workers",
        "prefetch_factor",
        "pin_memory",
        "val_pin_memory",
        "test_pin_memory",
    }
    filtered = {k: merged[k] for k in merged.keys() & allowed}
    return DataModuleCfg(**filtered)
