import os, sys
import numpy as np
import torch
import torch.distributed as dist
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lowspa_ddp.utils import get_parent_path

root = get_parent_path(lvl=1)

class GPTLoader():
    """
    Random contiguous-window data loader (DDP-friendly) that controls epoch length
    by either `tokens_per_epoch` or `steps_per_epoch`.

    Data format:
      - <split>.bin is a single 1D token stream stored as np.uint16.

    Each step:
      - Randomly pick a start index i.
      - x = data[i : i + block_size]
      - y = data[i + 1 : i + 1 + block_size]
      - Both returned as torch.long (CPU). Move to GPU at training time.

    DDP:
      - Start indices are sharded across ranks by modular arithmetic to ensure
        **no overlap across ranks within an epoch**.
      - Call `set_epoch(epoch)` at the start of each epoch to change RNG seed.

    Notes:
      - `batch_size` is per-rank.
      - Global tokens consumed per step ≈ batch_size * block_size * world_size.
      - If `tokens_per_epoch` is set, steps_per_epoch is derived from it;
        otherwise you can pass `steps_per_epoch` directly.
    """

    def __init__(self,
                 split: str = 'train',
                 batch_size: int = 64,
                 block_size: int = 256,
                 tokens_per_epoch: Optional[int] = None,
                 steps_per_epoch: Optional[int] = None,
                 seed: int = 1337,
                 pin_memory: bool = False):
        # Basic config
        dataset = 'openwebtext'
        self.data_dir = os.path.join(root, 'datasets', 'data', dataset)
        self.split = split
        self.batch_size = int(batch_size)
        self.block_size = int(block_size)
        self.path = os.path.join(self.data_dir, f"{split}.bin")
        self.pin_memory = pin_memory

        # DDP info (fallback to single-process if dist not initialized)
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        # Open the memory-mapped token file
        self._open_memmap()

        # Effective number of valid window start positions
        self.N_eff = self.N - self.block_size
        if self.N_eff <= 0:
            raise ValueError(f"Data length {self.N} must be greater than block_size {self.block_size}.")

        # Determine per-rank steps per epoch
        if steps_per_epoch is not None:
            self.steps_per_epoch = int(steps_per_epoch)
        elif tokens_per_epoch is not None:
            # Global tokens per step ≈ batch_size * block_size * world_size
            g_tokens_per_step = self.batch_size * self.block_size * self.world_size
            self.steps_per_epoch = max(int(tokens_per_epoch // g_tokens_per_step), 1)
        else:
            # Fallback: cover all possible starts once (often huge; not recommended for large corpora)
            self.steps_per_epoch = max(self.N_eff // (self.batch_size * self.world_size), 1)

        # RNG for reproducibility; updated via set_epoch()
        self.base_seed = int(seed)
        self._epoch = 0
        self._rng = torch.Generator().manual_seed(self.base_seed + self._epoch)

    def _open_memmap(self) -> None:
        """Open the token file as a read-only memmap (np.uint16)."""
        self._data = np.memmap(self.path, dtype=np.uint16, mode='r')
        self.N = len(self._data)

    def set_epoch(self, epoch: int) -> None:
        """
        Update RNG seed at the start of each epoch so all ranks draw a new (but consistent) sequence.
        The rank-based sharding still guarantees no overlap across ranks.
        """
        self._epoch = int(epoch)
        self._rng.manual_seed(self.base_seed + self._epoch)

    def __len__(self) -> int:
        """Per-rank number of steps in an epoch (lets tqdm infer total automatically)."""
        return self.steps_per_epoch

    def _rank_sharded_indices(self, k: int) -> torch.Tensor:
        """
        Sample k start indices in [0, N_eff) with rank-based modular sharding:
            i = rank + world_size * t,  where t ~ Uniform{0 .. max_t-1}
        This ensures different ranks never pick the same start index within the epoch.
        """
        # Number of "slots" per rank (ceil division)
        max_t = (self.N_eff + self.world_size - 1) // self.world_size

        # Draw t, then map to start indices i
        t = torch.randint(low=0, high=max_t, size=(k,), generator=self._rng)
        i = t * self.world_size + self.rank

        # Drop any out-of-range indices near the tail
        i = i[i < self.N_eff]

        # If we dropped some and have fewer than k, keep sampling until we have k
        while i.numel() < k:
            need = k - i.numel()
            t2 = torch.randint(low=0, high=max_t, size=(need,), generator=self._rng)
            i2 = t2 * self.world_size + self.rank
            i2 = i2[i2 < self.N_eff]
            i = torch.cat([i, i2], dim=0)

        return i[:k]

    def __iter__(self):
        """
        One iterator = one epoch.
        Yields `steps_per_epoch` batches, each with `batch_size` windows of length `block_size`.
        Returns CPU tensors (torch.long). Move to GPU in your training loop.
        """
        for _ in range(self.steps_per_epoch):
            # Choose per-rank, non-overlapping random start indices
            ix = self._rank_sharded_indices(self.batch_size)  # shape [B]

            # Build x, y on CPU (dtype long).
            # Simple, readable implementation; can be optimized if it becomes a bottleneck.
            x_list, y_list = [], []
            for i in ix.tolist():
                x_np = self._data[i : i + self.block_size].astype(np.int64, copy=True)
                y_np = self._data[i + 1 : i + 1 + self.block_size].astype(np.int64, copy=True)
                x_list.append(torch.from_numpy(x_np))
                y_list.append(torch.from_numpy(y_np))
            x = torch.stack(x_list)  # [B, T]
            y = torch.stack(y_list)

            if self.pin_memory:
                # Page-lock CPU memory to speed up subsequent H2D copies with non_blocking=True
                x = x.pin_memory()
                y = y.pin_memory()

            yield x, y

def get_gpt_dataloader(batch_size=128, 
                       block_size=1024):
    """Load GPT train/test dataloaders."""
    train_loader = GPTLoader(split='train',
                             batch_size=batch_size,
                             block_size=block_size,
                             steps_per_epoch=10)
    test_loader = GPTLoader(split='val',
                            batch_size=batch_size,
                            block_size=block_size)
    return train_loader, test_loader
