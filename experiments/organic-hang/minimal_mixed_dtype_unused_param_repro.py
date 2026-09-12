"""Probe FSDP2 unused-parameter zero-gradient dtype without pipeline parallelism.

Run with:
    torchrun --standalone --nproc-per-node=2 minimal_mixed_dtype_unused_param_repro.py
"""

from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


class ConditionalModel(nn.Module):
    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.always = nn.Linear(width, width)
        self.conditional = nn.Linear(width, width)

    def forward(self, value: torch.Tensor, use_conditional: bool) -> torch.Tensor:
        output = self.always(value)
        if use_conditional:
            output = output + self.conditional(value)
        return output


def main() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=30))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    try:
        model = ConditionalModel().to(device)
        fully_shard(
            model,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            ),
        )
        model.set_reduce_scatter_unused_params(True)

        value = torch.randn(4, 16, device=device, dtype=torch.bfloat16)
        loss = model(value, use_conditional=dist.get_rank() == 0).float().sum()
        loss.backward()
        dist.barrier()
        if dist.get_rank() == 0:
            print("completed without a mixed-gradient-dtype assertion")
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
