import atexit
import os

import torch
import torch.distributed as dist


def dist_is_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def dist_is_main_process() -> bool:
    if dist_is_enabled():
        return dist.get_rank() == 0
    return True


class EngineCfg:
    @classmethod
    def BATCH_DIM(cls) -> int:
        return int(os.environ.get("BATCH_DIM", 32))

    @classmethod
    def CHUNK_SIZE(cls) -> int:
        return int(os.environ.get("CHUNK_SIZE", 1024))

    @classmethod
    def NUM_WORKERS(cls) -> int:
        return int(os.environ.get("NUM_WORKERS", 1))

    @classmethod
    def DEVICE(cls) -> torch.device:
        return torch.device(
            f"""{os.environ.get("DEVICE", "cpu")}:{cls.TorchRun.LOCAL_RANK()}"""
        )

    class TorchRun:
        @classmethod
        def WORLD_SIZE(cls) -> int:
            return int(os.environ.get("WORLD_SIZE", 1))

        @classmethod
        def LOCAL_WORLD_SIZE(cls) -> int:
            return int(os.environ.get("LOCAL_WORLD_SIZE", 1))

        @classmethod
        def RANK(cls) -> int:
            return int(os.environ.get("RANK", 0))

        @classmethod
        def LOCAL_RANK(cls) -> int:
            return int(os.environ.get("LOCAL_RANK", 0))


if EngineCfg.TorchRun.WORLD_SIZE() > 1:
    dist.init_process_group("nccl", device_id=torch.device(EngineCfg.TorchRun.RANK()))
    atexit.register(dist.destroy_process_group)
