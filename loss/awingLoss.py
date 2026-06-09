import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F


class AWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1, use_weight_map=True):
        super(AWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        self.use_weight_map = use_weight_map
        # The piecewise coefficients A and C and the weight map depend only on
        # the ground-truth heatmap, which is identical across all model stages
        # within a batch. Cache them keyed on the ground-truth tensor identity
        # so the dense torch.pow / torch.log / max_pool2d work runs once per
        # batch instead of once per stage.
        self._gt_cache_ref = None
        self._gt_cache_meta = None
        self._gt_cache = None

    def __repr__(self):
        return "AWingLoss()"

    def generate_weight_map(self, heatmap, k_size=3, w=10):
        dilate = F.max_pool2d(heatmap, kernel_size=k_size, stride=1, padding=1)
        weight_map = torch.where(
            dilate < 0.2, torch.zeros_like(heatmap), torch.ones_like(heatmap)
        )
        return w * weight_map + 1

    def _ground_truth_terms(self, groundtruth):
        """Return (A, C, weight_map_or_None) for a ground-truth heatmap.

        These terms are pure functions of ``groundtruth``. When the target does
        not require grad (the normal training case) the result is cached and
        reused for repeated calls with the same tensor, e.g. across stacked
        model stages that all score against the same target.
        """
        meta = (groundtruth.shape, groundtruth.dtype, groundtruth.device)
        if (
            self._gt_cache is not None
            and self._gt_cache_meta == meta
            and self._gt_cache_ref is not None
            and self._gt_cache_ref() is groundtruth
        ):
            return self._gt_cache

        ratio = self.theta / self.epsilon
        expo = self.alpha - groundtruth
        # pow(theta/eps, alpha - gt) appears three times in the original; compute once.
        p = torch.pow(ratio, expo)
        one_plus_p = 1 + p
        # (theta/eps)^(expo - 1) == p / (theta/eps).
        A = self.omega * (1 / one_plus_p) * expo * (p / ratio) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(one_plus_p)
        weight = self.generate_weight_map(groundtruth) if self.use_weight_map else None

        terms = (A, C, weight)
        # Only cache constant targets; a grad-bearing target must not be
        # detached from the autograd graph via reuse.
        if not groundtruth.requires_grad:
            self._gt_cache_ref = weakref.ref(groundtruth)
            self._gt_cache_meta = meta
            self._gt_cache = terms
        else:
            self._gt_cache_ref = None
            self._gt_cache_meta = None
            self._gt_cache = None
        return terms

    def forward(self, output, groundtruth, batch_weights=None):
        """
        input:  b x n x h x w
        output: b x n x h x w => 1
        """
        A, C, weight = self._ground_truth_terms(groundtruth)
        delta = (output - groundtruth).abs()
        loss = torch.where(
            delta < self.theta,
            self.omega
            * torch.log(1 + torch.pow(delta / self.epsilon, self.alpha - groundtruth)),
            (A * delta - C),
        )
        if weight is not None:
            loss = loss * weight
        if batch_weights is None:
            return loss.mean()
        else:
            return (loss * batch_weights).mean()
