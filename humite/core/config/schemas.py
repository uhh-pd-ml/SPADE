from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DatasetConfig:
    # Identity and files
    name: str = ""
    data_dir: str = ""
    train_files: Dict[str, List[str]] = field(default_factory=dict)
    val_files: Dict[str, List[str]] = field(default_factory=dict)
    test_files: Dict[str, List[str]] = field(default_factory=dict)

    # Loader/file controls
    n_files_at_once: int = 1
    max_n_files_per_type: Optional[int] = None
    load_only_once: bool = False
    shuffle_only_once: bool = False
    shuffle_files: bool = True
    shuffle_data: bool = True
    seed: int = 44
    seed_shuffle_data: int = 2
    random_seed_for_per_file_shuffling: int = 42
    worker_row_sharding: bool = False
    row_shard_mode: str = "block"  # block | interleaved
    use_ram_cache: bool = True

    # Dataset/schema specifics
    dataset_type: str = "ECAL"
    sequence_sorting: str = "energy"  # energy | layer | layer_energy | random | none
    n_showers: Optional[int] = None

    # Geometry and padding
    pad_length: int = 1700
    nbins_x: int = 30
    nbins_y: int = 30
    nbins_z: int = 30
    nbins_total: int = 27000

    # Tokenization and modes
    energy_threshold: float = 0.0
    use_energy_tokenization: bool = False
    combine_xyz_tokens: bool = False
    combine_xyz: bool = False
    hierarchical: bool = True
    axis_mode: str = "split"  # split | combined

    # Vocab and energy transform
    vocab_size_x: Optional[int] = None
    vocab_size_y: Optional[int] = None
    vocab_size_z: Optional[int] = None
    vocab_size_total: Optional[int] = None
    energy_vocab_size: int = 1000
    energy_min: float = 0.01
    energy_max: float = 40.0
    energy_log_scale: bool = True

    # HCal resolution binning (active only when use_energy_tokenization=True
    # and energy_binning_mode="hcal_resolution"). energy_vocab_size is then
    # derived from the generated edges (num_bins + 2) and overrides the field above.
    energy_binning_mode: str = "uniform"  # "uniform" | "hcal_resolution"
    energy_hcal_A: float = 0.50
    energy_hcal_B: float = 0.0
    energy_hcal_C: float = 0.02
    energy_hcal_step_sigma: float = 0.5
    energy_hcal_e_min: float = 0.001  # GeV
    energy_hcal_e_max: float = 150.0  # GeV

    # Optional 10-80-10 label smoothing across vocab dim only
    energy_label_smoothing: str = "none"  # "none" | "10-80-10"

    # Optional raw features block (for legacy preprocessor compatibility)
    features: Dict[str, object] = field(default_factory=dict)


@dataclass
class TransformerCfg:
    embedding_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    attn_dropout: float = 0.0
    max_seq_len: int = 1024
    bias: bool = True
    use_parallel_residual: bool = False


@dataclass
class ModelConfig:
    name: str = "hybrid"
    transformer: TransformerCfg = field(default_factory=TransformerCfg)
    kwargs: Dict[str, object] = field(default_factory=dict)


@dataclass
class TrainerConfig:
    batch_size: int = 2
    max_steps: int = 5
    num_sanity_val_steps: int = 0
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    generative_eval: bool = False
    num_workers: int = 0
    val_num_workers: Optional[int] = None
    test_num_workers: Optional[int] = None
    pin_memory: bool = True
    val_pin_memory: Optional[bool] = None
    test_pin_memory: Optional[bool] = None
