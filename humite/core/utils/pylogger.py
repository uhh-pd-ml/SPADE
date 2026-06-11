import logging
import os
import os as _os

try:
    import torch.distributed as _dist  # type: ignore
except Exception:  # noqa: BLE001
    _dist = None  # type: ignore


def get_pylogger(name=__name__, rank=None) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger.

    Parameters
    ----------
    name : str, optional
        Name of the logger. Default is __name__.
    rank : int, optional
        Rank of the current process. If not provided, it will be retrieved from
        torch.distributed.get_rank().

    Returns
    -------
    logging.Logger
        Logger object.
    """
    if rank is None:
        r = None
        try:
            if _dist is not None and _dist.is_available() and _dist.is_initialized():
                r = int(_dist.get_rank())
        except Exception:  # noqa: BLE001
            r = None
        if r is None:
            for k in ("RANK", "WORLD_RANK", "SLURM_PROCID"):
                v = _os.getenv(k)
                if v is not None:
                    try:
                        r = int(v)
                        break
                    except Exception:  # noqa: BLE001
                        continue
        rank = r if r is not None else "unknown"
    rank_string = f"rank:{rank}"

    hostname = os.getenv("HOSTNAME", default="unknown-host")

    logger = logging.getLogger(f"{hostname}|{rank_string}|{name}")

    return logger
