import os

import torch


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
        return torch.device(f"""{os.environ.get("DEVICE", "cpu")}:0""")
