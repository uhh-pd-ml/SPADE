import glob
import os
import random
from typing import Generator, Iterable, List

import awkward as ak
import torch
import torch.distributed as dist
from torch.distributed import get_rank, get_world_size
from torch.utils.data import IterableDataset, get_worker_info

from humite.core.config.schemas import DatasetConfig
from humite.core.data.datasets.io import (
    get_h5_row_count,
    read_shower_file_and_energy,
    standardize_shower_schema,
)
from humite.core.data.geometry import get_spatial_field_names
from humite.core.data.preprocessing.adapters import build_adapter
from humite.core.data.preprocessing.sorters import sort_showers
from humite.core.utils.pylogger import get_pylogger


class CalorimeterIterableDataset(IterableDataset):
    def __init__(self, files_dict, dataset: DatasetConfig, repeat: bool = False):
        self.log = get_pylogger(__name__)
        self.dataset = dataset
        self.adapter = build_adapter(dataset)
        self.raw_spatial_fields = get_spatial_field_names(dataset.dataset_type)
        self.n_showers = getattr(dataset, "n_showers", None)
        self.max_n_files_per_type = getattr(dataset, "max_n_files_per_type", None)
        self.sequence_sorting = getattr(dataset, "sequence_sorting", "energy")
        self.n_files_at_once = int(getattr(dataset, "n_files_at_once", 1) or 1)
        self.shuffle_files = bool(getattr(dataset, "shuffle_files", True))
        self.shuffle_data = bool(getattr(dataset, "shuffle_data", True))
        self.seed = int(getattr(dataset, "seed", 44))
        self.seed_shuffle_data = int(getattr(dataset, "seed_shuffle_data", 2))
        self.load_only_once = bool(getattr(dataset, "load_only_once", False))
        self.shuffle_only_once = bool(getattr(dataset, "shuffle_only_once", False))
        self.random_seed_for_per_file_shuffling = int(
            getattr(dataset, "random_seed_for_per_file_shuffling", 42)
        )
        self.worker_row_sharding = bool(getattr(dataset, "worker_row_sharding", False))
        self.row_shard_mode = str(getattr(dataset, "row_shard_mode", "block"))
        self.use_ram_cache = bool(getattr(dataset, "use_ram_cache", False))
        self.repeat = bool(repeat)

        self.files_by_type = self._expand_and_cap_files(files_dict)
        self.file_list: List[str] = [f for files in self.files_by_type.values() for f in files]
        self.n_types = len(self.files_by_type)
        self.n_files_at_once = (
            self.n_types if self.n_files_at_once is None else self.n_files_at_once
        )
        if self.n_files_at_once > max(1, len(self.file_list)):
            self.n_files_at_once = len(self.file_list)

        self._epoch_file_counter = 0
        self._cached_events_for_rank = None
        self._file_events_cache: dict[tuple, list] = {}
        self._iterator_index = 0
        self._rank_row_shard: tuple[int, int] | None = None  # (rank, world) when sharing files
        self._has_logged_shard_config = False

    def _compute_worker_row_slice(
        self, total_rows: int, worker_id: int, num_workers: int
    ) -> tuple[int, int, int] | None:
        if total_rows <= 0:
            return None
        if num_workers <= 1:
            return (0, total_rows, 1)
        if self.row_shard_mode == "interleaved":
            if worker_id >= total_rows:
                return None
            return (worker_id, total_rows, num_workers)

        rows_per_worker = total_rows // num_workers
        remainder = total_rows % num_workers
        start = worker_id * rows_per_worker + min(worker_id, remainder)
        length = rows_per_worker + (1 if worker_id < remainder else 0)
        if length <= 0:
            return None
        return (start, start + length, 1)

    def _load_file(self, filename):
        try:
            worker = get_worker_info()
            worker_id = int(getattr(worker, "id", 0)) if worker is not None else 0
            num_workers = int(getattr(worker, "num_workers", 1)) if worker is not None else 1
        except Exception:
            worker_id = 0
            num_workers = 1
        try:
            rank_id = get_rank() if dist.is_initialized() else 0
        except Exception:
            rank_id = 0
        load_desc = "full_file"
        cache_key = None

        load_kwargs = {
            "n_load": self.n_showers,
            "dataset_type": getattr(self.dataset, "dataset_type", None),
        }

        # When files are shared across ranks, compute rank-level row range first
        rank_row_start = 0
        rank_row_count = None  # None = full file
        if self._rank_row_shard is not None:
            rank, world = self._rank_row_shard
            total_rows = int(get_h5_row_count(filename))
            if self.n_showers is not None:
                total_rows = min(total_rows, int(self.n_showers))
            rows_per_rank = total_rows // world
            remainder = total_rows % world
            rank_row_start = rank * rows_per_rank + min(rank, remainder)
            rank_row_count = rows_per_rank + (1 if rank < remainder else 0)
            if rank_row_count <= 0:
                self.log.debug(
                    f"rank={rank_id} worker={worker_id} skipping {filename} (no rows for rank)"
                )
                return []

        if self.worker_row_sharding:
            if rank_row_count is not None:
                effective_total = rank_row_count
            else:
                effective_total = int(get_h5_row_count(filename))
                if self.n_showers is not None:
                    effective_total = min(effective_total, int(self.n_showers))
            shard = self._compute_worker_row_slice(effective_total, worker_id, num_workers)
            if shard is None:
                self.log.debug(
                    f"rank={rank_id} worker={worker_id} skipping {filename} (no rows assigned)"
                )
                return []
            w_start, w_stop, w_step = shard
            start = rank_row_start + w_start
            stop = rank_row_start + w_stop
            step = w_step
            load_kwargs = {
                "n_load": None,
                "start": int(start),
                "stop": int(stop),
                "step": int(step),
                "dataset_type": getattr(self.dataset, "dataset_type", None),
            }
            file_total = int(get_h5_row_count(filename))
            load_desc = f"rows[{start}:{stop}:{step}]/{file_total}"
            cache_key = (filename, int(start), int(stop), int(step), "shard")
        elif rank_row_count is not None:
            # Rank-level sharding without worker row sharding
            start = rank_row_start
            stop = rank_row_start + rank_row_count
            load_kwargs = {
                "n_load": None,
                "start": int(start),
                "stop": int(stop),
                "step": 1,
                "dataset_type": getattr(self.dataset, "dataset_type", None),
            }
            file_total = int(get_h5_row_count(filename))
            load_desc = f"rows[{start}:{stop}:1]/{file_total}"
            cache_key = (filename, int(start), int(stop), 1, "shard")
        else:
            cache_key = (
                filename,
                int(self.n_showers) if self.n_showers is not None else None,
                "full",
            )

        if self.use_ram_cache and cache_key is not None and cache_key in self._file_events_cache:
            return self._file_events_cache[cache_key]

        self.log.debug(f"rank={rank_id} worker={worker_id} loading {filename} {load_desc}")

        showers, energies = read_shower_file_and_energy(filename, **load_kwargs)
        showers = standardize_shower_schema(showers, dataset_type=self.dataset.dataset_type)

        showers = sort_showers(
            showers,
            self.sequence_sorting,
            spatial_fields=self.raw_spatial_fields,
        )

        showers = self.adapter.preprocess(showers, energies)
        n_events = len(showers["energy"])
        events = []
        spatial_axis_names = tuple(self.raw_spatial_fields)
        for i in range(n_events):
            event = {
                "energy": showers["energy"][i],
                "incident_energy": showers["incident_energy"][i],
                "n_hits": showers["n_hits"][i],
                "spatial_axis_names": spatial_axis_names,
                "hit_mask": showers["hit_mask"][i],
            }
            for axis_name in spatial_axis_names:
                event[axis_name] = showers[axis_name][i]
            events.append(event)
        try:
            worker = get_worker_info()
            is_worker0 = (worker is None) or int(getattr(worker, "id", 0)) == 0
            is_rank0 = (not dist.is_initialized()) or get_rank() == 0
            if is_worker0 and is_rank0:
                self.log.info(f"Loaded {n_events} events from {filename}")
                self.log.debug(f"Event keys: {list(events[0].keys()) if n_events > 0 else 'N/A'}")
        except Exception:
            pass

        if self.use_ram_cache and cache_key is not None:
            self._file_events_cache[cache_key] = events

        return events

    def _get_rank_world(self) -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return int(get_rank()), int(get_world_size())
        rank_env = int(os.environ.get("RANK", "0"))
        world_env = int(os.environ.get("WORLD_SIZE", "1"))
        return rank_env, max(1, world_env)

    def _get_files_for_this_rank(self, seed_for_files: int):
        files = list(self.file_list)
        if self.shuffle_files:
            random.Random(seed_for_files).shuffle(files)
        rank, world = self._get_rank_world()
        self._rank_row_shard = None
        if world > 1:
            if len(files) >= world:
                n_files_usable_for_balanced_ranks = (len(files) // world) * world
                files = files[:n_files_usable_for_balanced_ranks]
                files = [
                    file_path
                    for file_index, file_path in enumerate(files)
                    if file_index % world == rank
                ]
            else:
                # Fewer files than ranks: all ranks read all files,
                # each rank reads a distinct row slice per file.
                self._rank_row_shard = (rank, world)
                self.log.info(
                    f"Fewer files ({len(files)}) than ranks ({world}). "
                    f"Using rank-level row sharding (rank={rank}/{world})."
                )
        worker_info = get_worker_info()
        if worker_info is not None and not self.worker_row_sharding:
            worker_id = int(worker_info.id)
            num_workers = int(worker_info.num_workers)
            files = [
                file_path
                for file_index, file_path in enumerate(files)
                if file_index % num_workers == worker_id
            ]
        return files

    def _expand_and_cap_files(self, files_dict) -> dict:
        expanded = {}
        for k, patterns in files_dict.items():
            coll: List[str] = []
            for p in patterns:
                coll.extend(sorted(glob.glob(p)))
            if self.max_n_files_per_type is not None:
                coll = coll[: self.max_n_files_per_type]
            expanded[k] = coll
        return expanded

    def _iterate_file_batches(self, files: List[str]) -> Iterable[List[str]]:
        nfa = int(self.n_files_at_once) if self.n_files_at_once else 0
        if nfa <= 0:
            return
        for i in range(0, len(files), nfa):
            yield files[i : i + nfa]

    def __iter__(self):
        worker_info = get_worker_info()
        iterator_seed = int(self.seed + self._iterator_index)
        self._iterator_index += 1

        if not self._has_logged_shard_config:
            try:
                rank_id, world = self._get_rank_world()
                wid = int(getattr(worker_info, "id", 0)) if worker_info is not None else 0
                nworkers = (
                    int(getattr(worker_info, "num_workers", 1)) if worker_info is not None else 1
                )
                if wid == 0:
                    self.log.info(
                        "Dataset shard config: "
                        f"rank={rank_id}/{world}, worker_row_sharding={self.worker_row_sharding}, "
                        f"row_shard_mode={self.row_shard_mode}, num_workers={nworkers}, "
                        f"repeat={self.repeat}, n_files_total={len(self.file_list)}"
                    )
            except Exception:
                pass
            self._has_logged_shard_config = True

        if self.load_only_once and self._cached_events_for_rank is not None:
            cycle_index = 0
            while True:
                events = list(self._cached_events_for_rank)
                if self.shuffle_data and not self.shuffle_only_once:
                    rng = random.Random(self.seed_shuffle_data + iterator_seed + cycle_index)
                    rng.shuffle(events)
                for e in events:
                    yield e
                if not self.repeat:
                    return
                cycle_index += 1

        files_this_rank = self._get_files_for_this_rank(seed_for_files=iterator_seed + self.seed)
        cached_events_for_rank = [] if self.load_only_once else None
        cycle_index = 0
        while True:
            for batch_files in self._iterate_file_batches(files_this_rank):
                batch_events = []
                for f in batch_files:
                    file_events = self._load_file(f)
                    batch_events.extend(file_events)

                if self.shuffle_data:
                    rng = random.Random(self.seed_shuffle_data + iterator_seed + cycle_index)
                    rng.shuffle(batch_events)

                if self.load_only_once:
                    cached_events_for_rank.extend(batch_events)

                for e in batch_events:
                    yield e

                self._epoch_file_counter += len(batch_files)

            if self.load_only_once:
                self._cached_events_for_rank = list(cached_events_for_rank)

            if not self.repeat:
                return
            cycle_index += 1
