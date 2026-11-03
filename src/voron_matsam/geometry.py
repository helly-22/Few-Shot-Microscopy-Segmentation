"""Soft power diagrams. Coordinates are normalized (x, y) in [0, 1]."""

import torch
from torch import Tensor


def soft_voronoi(centers: Tensor, height: int, width: int,
                 temperature: float = 0.1, weights: Tensor | None = None) -> Tensor:
    """Return [B, K, H, W] probabilities; positive weights expand a cell.

    K must be provided by a seed predictor. This layer does not infer seed count.
    All seeds participate everywhere: background/ROI handling belongs to the caller.
    """
    if centers.ndim != 3 or centers.shape[-1] != 2 or centers.shape[1] == 0:
        raise ValueError("centers must have shape [B, K, 2] with K > 0")
    if not centers.is_floating_point() or not torch.isfinite(centers).all():
        raise ValueError("centers must be finite floating point coordinates")
    if height < 1 or width < 1 or not 0 < temperature < float("inf"):
        raise ValueError("positive image dimensions and finite temperature required")
    y, x = torch.meshgrid(
        torch.linspace(0, 1, height, device=centers.device, dtype=centers.dtype),
        torch.linspace(0, 1, width, device=centers.device, dtype=centers.dtype),
        indexing="ij",
    )
    grid = torch.stack((x, y), dim=-1)
    scores = -(grid[None, None] - centers[:, :, None, None]).square().sum(-1)
    if weights is not None:
        if weights.shape != centers.shape[:2] or not torch.isfinite(weights).all():
            raise ValueError("weights must be finite with shape [B, K]")
        scores = scores + weights[:, :, None, None]
    return (scores / temperature).softmax(dim=1)


def voronoi_consistency_loss(logits: Tensor, centers: Tensor,
                            temperature: float = 0.1,
                            weights: Tensor | None = None) -> Tensor:
    """Symmetric agreement between aligned instance channels and soft cells.

    Both branches receive gradients. Channels must correspond to the same seeds;
    never compare arbitrary instance ID orderings without matching them first.
    """
    if logits.ndim != 4 or logits.shape[:2] != centers.shape[:2]:
        raise ValueError("logits [B, K, H, W] must match centers [B, K, 2]")
    cells = soft_voronoi(centers, *logits.shape[-2:], temperature, weights)
    return (logits.softmax(dim=1) - cells).square().sum(dim=1).mean()
