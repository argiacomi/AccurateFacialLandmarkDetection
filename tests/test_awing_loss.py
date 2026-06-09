"""Equivalence and caching guards for the AWing heatmap loss.

The forward pass was refactored to deduplicate the ``pow(theta/eps, alpha-gt)``
term and to cache the ground-truth-only coefficients (A, C, weight map) across
repeated calls with the same target, e.g. across stacked model stages. These
tests lock in numerical equivalence to a verbatim reference and the cache's
reuse/invalidation behaviour.
"""

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import AWingLoss


class _ReferenceAWingLoss(nn.Module):
    """Verbatim copy of the original AWingLoss.forward for comparison."""

    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1, use_weight_map=True):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        self.use_weight_map = use_weight_map

    def generate_weight_map(self, heatmap, k_size=3, w=10):
        dilate = F.max_pool2d(heatmap, kernel_size=k_size, stride=1, padding=1)
        weight_map = torch.where(
            dilate < 0.2, torch.zeros_like(heatmap), torch.ones_like(heatmap)
        )
        return w * weight_map + 1

    def forward(self, output, groundtruth, batch_weights=None):
        delta = (output - groundtruth).abs()
        A = (
            self.omega
            * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - groundtruth)))
            * (self.alpha - groundtruth)
            * (torch.pow(self.theta / self.epsilon, self.alpha - groundtruth - 1))
            * (1 / self.epsilon)
        )
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, self.alpha - groundtruth)
        )
        loss = torch.where(
            delta < self.theta,
            self.omega
            * torch.log(1 + torch.pow(delta / self.epsilon, self.alpha - groundtruth)),
            (A * delta - C),
        )
        if self.use_weight_map:
            loss = loss * self.generate_weight_map(groundtruth)
        if batch_weights is None:
            return loss.mean()
        return (loss * batch_weights).mean()


def test_forward_matches_reference_for_all_modes():
    torch.manual_seed(0)
    for use_weight_map in (True, False):
        for with_batch_weights in (False, True):
            ref = _ReferenceAWingLoss(use_weight_map=use_weight_map)
            new = AWingLoss(use_weight_map=use_weight_map)
            output = torch.rand(4, 68, 16, 16)
            groundtruth = torch.rand(4, 68, 16, 16)
            bw = torch.rand(4, 68, 1, 1) if with_batch_weights else None
            assert torch.allclose(
                ref(output, groundtruth, bw), new(output, groundtruth, bw), atol=1e-5
            )


def test_gradient_wrt_output_matches_reference():
    torch.manual_seed(1)
    groundtruth = torch.rand(4, 98, 16, 16)
    out_ref = torch.rand(4, 98, 16, 16, requires_grad=True)
    out_new = out_ref.detach().clone().requires_grad_()

    _ReferenceAWingLoss()(out_ref, groundtruth).backward()
    AWingLoss()(out_new, groundtruth).backward()

    assert torch.allclose(out_ref.grad, out_new.grad, atol=1e-5)


def test_cached_terms_reused_across_stages_match_per_stage_reference():
    torch.manual_seed(2)
    new = AWingLoss()
    ref = _ReferenceAWingLoss()
    shared_gt = torch.rand(4, 68, 16, 16)
    for _ in range(8):  # mimic nstack stages scoring against one target
        output = torch.rand(4, 68, 16, 16)
        assert torch.allclose(new(output, shared_gt), ref(output, shared_gt), atol=1e-5)


def test_cache_invalidates_when_groundtruth_changes():
    torch.manual_seed(3)
    new = AWingLoss()
    ref = _ReferenceAWingLoss()
    output = torch.rand(2, 29, 8, 8)

    gt_a = torch.rand(2, 29, 8, 8) + 5.0
    assert torch.allclose(new(output, gt_a), ref(output, gt_a), atol=1e-5)

    gt_b = torch.zeros(2, 29, 8, 8) + 0.01
    assert torch.allclose(new(output, gt_b), ref(output, gt_b), atol=1e-5)


def test_cache_does_not_serve_stale_result_after_target_freed():
    new = AWingLoss()
    ref = _ReferenceAWingLoss()
    output = torch.rand(2, 29, 8, 8)

    gt_a = torch.rand(2, 29, 8, 8) + 5.0
    new(output, gt_a)
    del gt_a
    gc.collect()

    gt_b = torch.zeros(2, 29, 8, 8) + 0.01
    assert torch.allclose(new(output, gt_b), ref(output, gt_b), atol=1e-5)


def test_grad_bearing_target_is_not_cached():
    new = AWingLoss()
    gt = torch.rand(2, 29, 8, 8, requires_grad=True)
    output = torch.rand(2, 29, 8, 8)
    new(output, gt)
    assert new._gt_cache is None
