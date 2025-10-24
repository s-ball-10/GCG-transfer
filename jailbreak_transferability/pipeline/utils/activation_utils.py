"""Subset‑aware loader for prompt & jailbreak activations (hour‑level runtime)

This version loads **only the k×k block** of jailbreak activations corresponding
to `keep_ids`, eliminating the O(N²) blow‑up.  Wall‑time now scales roughly
with `k²` instead of `N²`, enabling repeated hourly runs.

Key features
------------
* Two‑pass streaming:
  1. **Metadata pass** computes total rows → true *N*.
  2. **Data pass** copies only required rows into a `(k², L, d)` buffer.
* Works with the existing chunk layout (consecutive rows of the flattened
  square tensor).
* Preserves all env‑vars introduced earlier (`JBT_DEVICE`, `JBT_NUM_WORKERS`,
  `JBT_DTYPE`, `JBT_DISK_DTYPE`).

Drop‑in API compatibility is maintained.
"""
from __future__ import annotations

import math
import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple, Union
from tqdm import tqdm

import torch
from pipeline.utils import utils

__all__ = [
    "JBView",
    "get_prompt_activations",
    "get_jailbreak_view",
]

###############################################################################
# Device & dtype config -------------------------------------------------------
###############################################################################

def _select_device() -> torch.device:
    env = os.getenv("JBT_DEVICE")
    return torch.device(env) if env else torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = _select_device()

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
DEFAULT_DTYPE = _DTYPE_MAP.get(os.getenv("JBT_DTYPE", "fp16"), torch.float16)
STAGING_DTYPE = _DTYPE_MAP.get(os.getenv("JBT_DISK_DTYPE", "fp16"), torch.float16)

Index = Union[int, slice, torch.Tensor, List[int], Tuple[int, ...]]

###############################################################################
# Finite‑value assertion ------------------------------------------------------
###############################################################################

def _assert_finite(t: torch.Tensor, msg: str) -> None:
    if not os.getenv("JBT_SKIP_ASSERT") and not torch.isfinite(t).all():
        raise ValueError(f"Non‑finite values detected: {msg}")

###############################################################################
# JBView: tensor‑like accessor -------------------------------------------------
###############################################################################
class JBView(torch.nn.Module):
    """Read‑only view that preserves original prompt/suffix IDs."""

    def __init__(self, compact: torch.Tensor, keep_ids: Sequence[int]):
        super().__init__()
        _assert_finite(compact, "jb_compact @ JBView init")
        self.register_buffer("_t", compact, persistent=False)
        self._id2idx: Dict[int, int] = {int(pid): i for i, pid in enumerate(keep_ids)}

    def _remap(self, key: Index) -> Index:
        tbl = self._id2idx
        if isinstance(key, slice):
            return key
        if isinstance(key, int):
            return tbl[key]
        if isinstance(key, torch.Tensor):
            mapped = torch.as_tensor([tbl[int(k)] for k in key.flatten()], dtype=torch.long, device=self._t.device)
            return mapped.view(key.shape)
        if isinstance(key, (list, tuple)):
            return [tbl[int(k)] for k in key]
        raise TypeError("Unsupported index type for JBView")

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (4 - len(key))
        p, s, l, d = key
        return self._t[self._remap(p), self._remap(s), l, d]

    @property
    def shape(self):
        return self._t.shape

    def size(self, *dims):
        return self._t.size(*dims)

    def __len__(self):
        return self._t.size(0)

    def _as_tensor(self):
        return self._t

    def __sub__(self, other):
        other_t = other._as_tensor() if isinstance(other, JBView) else other
        return self._t - other_t

    def __rsub__(self, other):
        other_t = other._as_tensor() if isinstance(other, JBView) else other
        return other_t - self._t

###############################################################################
# CPU loading helpers ---------------------------------------------------------
###############################################################################

def _load_chunk(path: str) -> torch.Tensor:
    """Load a .pt/.pth chunk and cast to STAGING_DTYPE."""
    t = torch.load(path, map_location="cpu", weights_only=True)
    return t.to(dtype=STAGING_DTYPE)


def _enumerate_chunks(jb_dir: str):
    """Yield (path, num_rows) for each chunk in sorted order."""
    paths = [osp.join(jb_dir, f) for f in sorted(os.listdir(jb_dir)) if f.endswith((".pt", ".pth"))]
    if not paths:
        raise FileNotFoundError(f"No tensor chunks in {jb_dir}")
    for p in paths:
        t_meta = torch.load(p, map_location="cpu", weights_only=True)
        yield p, t_meta.shape[0], t_meta.shape[1:]


def _subset_jb_compact_cpu(jb_dir: str, keep_ids: Sequence[int]) -> torch.Tensor:
    for path in sorted(os.listdir(jb_dir)):
        if not path.endswith((".pt", ".pth")): continue
        full = torch.load(osp.join(jb_dir, path), map_location="cpu", weights_only=True)
        n_nans  = torch.isnan(full).sum().item()
        n_infs  = torch.isinf(full).sum().item()
        if n_nans or n_infs:
            print(f"[CORRUPT] {path}: {n_nans} NaNs, {n_infs} Infs")

    """Stream only the (k×k) rows/cols defined by keep_ids into a compact tensor."""
    # ── Pass 1: gather chunk metadata & infer N ──────────────────────────────
    chunk_meta: List[Tuple[str, int, int, int]] = []  # (path, rows, start_row, end_row)
    total_rows = 0
    sample_shape = None

    # Build sorted list of chunk paths
    paths = [osp.join(jb_dir, f) for f in sorted(os.listdir(jb_dir)) if f.endswith((".pt", ".pth"))]
    total_chunks = len(paths)
    # Show actual progress bar for metadata
    for path in tqdm(paths, desc="Gathering chunk metadata", total=total_chunks, ascii=True, dynamic_ncols=True):
        t_meta = torch.load(path, map_location="cpu", weights_only=True)
        rows = t_meta.shape[0]
        shp = t_meta.shape[1:]
        if sample_shape is None:
            sample_shape = shp
        chunk_meta.append((path, rows, total_rows, total_rows + rows - 1))
        total_rows += rows

    N = int(math.isqrt(total_rows))
    if N * N != total_rows:
        raise ValueError("Jailbreak flat tensor is not a perfect square → cannot infer N")

    # ── Build lookup table for required global rows ─────────────────────────
    k = len(keep_ids)
    dest_map: Dict[int, int] = {}
    needed_rows: set[int] = set()
    dest_row = 0
    for pid in keep_ids:
        for sid in keep_ids:
            g = pid * N + sid
            dest_map[g] = dest_row
            needed_rows.add(g)
            dest_row += 1

    # Pre‑allocate destination buffer (flattened k² rows)
    dest = torch.empty((k * k, *sample_shape), dtype=STAGING_DTYPE)

    # ── Helper that conditionally loads a chunk ─────────────────────────────
    def _maybe_load_and_copy(path: str, rows: int, start_row: int, end_row: int):
        has_overlap = any(r in needed_rows for r in range(start_row, end_row + 1))
        if not has_overlap:
            return
        chunk = _load_chunk(path)
        local_rows: List[int] = []
        dest_rows: List[int] = []
        for idx in range(rows):
            g_row = start_row + idx
            if g_row in needed_rows:
                local_rows.append(idx)
                dest_rows.append(dest_map[g_row])
        if local_rows:
            src_idx = torch.tensor(local_rows, dtype=torch.long)
            dst_idx = torch.tensor(dest_rows, dtype=torch.long)
            dest.index_copy_(0, dst_idx, chunk.index_select(0, src_idx))
        del chunk

    # ── Pass 2: load required rows ──────────────────────────────────────────
    num_workers = int(os.getenv("JBT_NUM_WORKERS", "0"))
    if num_workers > 0 and len(chunk_meta) > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            for _ in tqdm(ex.map(lambda meta: _maybe_load_and_copy(*meta), chunk_meta), total=len(chunk_meta), desc="Loading chunks", ascii=True, dynamic_ncols=True):
                pass
    else:
        for meta in tqdm(chunk_meta, desc="Loading chunks", ascii=True, dynamic_ncols=True):
            _maybe_load_and_copy(*meta)

    _assert_finite(dest, "jb_subset compact CPU")
    return dest.view(k, k, *sample_shape)

###############################################################################
# Cached builder --------------------------------------------------------------
###############################################################################

@lru_cache(maxsize=4)
@torch.no_grad()
def _build_tensors(prompt_path: str, jb_dir: str, keep_ids: Tuple[int, ...]):
    keep_cpu = torch.tensor(keep_ids, dtype=torch.long)

    # ── Prompt activations (simple 1‑D slice) ────────────────────────────────
    prompt_full = torch.load(prompt_path, map_location="cpu", weights_only=True).to(dtype=STAGING_DTYPE)
    prompt_compact_cpu = prompt_full.index_select(0, keep_cpu)

    # ── Jailbreak activations (subset loader) ────────────────────────────────
    jb_compact_cpu = _subset_jb_compact_cpu(jb_dir, keep_ids)

    # ── Host→device transfer ─────────────────────────────────────────────────
    to_dev = (lambda x: x.pin_memory().to(DEVICE, dtype=DEFAULT_DTYPE, non_blocking=True)) \
        if DEVICE.type == "cuda" else (lambda x: x.to(dtype=DEFAULT_DTYPE))

    prompt_compact = to_dev(prompt_compact_cpu)
    jb_compact = to_dev(jb_compact_cpu)

    return prompt_compact, jb_compact, keep_cpu.to(DEVICE)

###############################################################################
# Public API ------------------------------------------------------------------
###############################################################################

def _fingerprint(cfg) -> Tuple[str, str]:
    return cfg.prompt_activations_path(), cfg.jailbreak_activations_dir()

def get_prompt_activations(cfg, device: str | torch.device | None = None) -> torch.Tensor:
    dev = torch.device(device) if device else DEVICE
    prompt_path, jb_dir = _fingerprint(cfg)
    keep = tuple(utils.get_previously_refused_indices(cfg))
    prompt_compact, _, _ = _build_tensors(prompt_path, jb_dir, keep)
    return prompt_compact.to(dev, non_blocking=True) if dev != DEVICE else prompt_compact

def get_jailbreak_view(cfg, device: str | torch.device | None = None) -> JBView:
    dev = torch.device(device) if device else DEVICE
    prompt_path, jb_dir = _fingerprint(cfg)
    keep = tuple(utils.get_previously_refused_indices(cfg))
    _, jb_compact, _ = _build_tensors(prompt_path, jb_dir, keep)
    jb_compact = jb_compact.to(dev, non_blocking=True) if dev != DEVICE else jb_compact
    return JBView(jb_compact, keep)
