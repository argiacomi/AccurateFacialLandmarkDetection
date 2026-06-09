import torch
import torch.nn as nn
import torch.nn.functional as F

from .smoothL1Loss import SmoothL1Loss
from .wingLoss import WingLoss


def get_channel_sum(input):
    temp = torch.sum(input, dim=3)
    output = torch.sum(temp, dim=2)
    return output


def expand_two_dimensions_at_end(input, dim1, dim2):
    input = input.unsqueeze(-1).unsqueeze(-1)
    input = input.expand(-1, -1, dim1, dim2)
    return input


class STARLoss_v2(nn.Module):
    def __init__(
        self,
        w=1,
        dist="smoothl1",
        num_dim_image=2,
        EPSILON=1e-5,
        softmax_normalized=True,
        check_finite=False,
        check_finite_interval=0,
    ):
        super(STARLoss_v2, self).__init__()
        self.w = w
        self.num_dim_image = num_dim_image
        self.EPSILON = EPSILON
        self.dist = dist
        # NaN/Inf guard on the covariance before eigh. Off by default because
        # `torch.isfinite(...).all()` consumed in a Python `if` forces a CUDA
        # host-device sync every forward. Enable for debugging/CI/smoke runs.
        # check_finite_interval > 0 runs the guard only every N eigh calls.
        self.check_finite = bool(check_finite)
        self.check_finite_interval = int(check_finite_interval)
        self._eigh_step = 0
        if self.dist == "smoothl1":
            self.dist_func = SmoothL1Loss()
        elif self.dist == "l1":
            self.dist_func = F.l1_loss
        elif self.dist == "l2":
            self.dist_func = F.mse_loss
        elif self.dist == "wing":
            self.dist_func = WingLoss()
        else:
            raise NotImplementedError
        self.softmax_normalized = softmax_normalized

        # Runtime-only cache. Not a buffer because it is keyed by dynamic
        # (height, width, device, dtype) and should not appear in state_dict.
        self._grid_cache = {}

    def __repr__(self):
        return "STARLoss_v2()"

    def _work_dtype(self, tensor: torch.Tensor) -> torch.dtype:
        # Keep STAR statistics/eigensolve in fp32 under AMP. Many CUDA builds do
        # not support stable eigh for fp16/bfloat16.
        if tensor.dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return tensor.dtype

    def _grid_cache_key(self, h, w, *, device, dtype):
        device = torch.device(device)
        return (
            int(h),
            int(w),
            device.type,
            device.index,
            str(dtype),
        )

    def _make_grid(self, h, w, *, device=None, dtype=None):
        """Return cached normalized yy/xx grids on the requested device/dtype."""

        device = torch.device("cpu" if device is None else device)
        dtype = torch.float32 if dtype is None else dtype
        key = self._grid_cache_key(h, w, device=device, dtype=dtype)

        cached = self._grid_cache.get(key)
        if cached is not None:
            yy, xx = cached
            if yy.device == device and xx.device == device and yy.dtype == dtype:
                return yy, xx

        yy, xx = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype) / max(int(h) - 1, 1),
            torch.arange(w, device=device, dtype=dtype) / max(int(w) - 1, 1),
            indexing="ij",
        )
        self._grid_cache[key] = (yy, xx)
        return yy, xx

    def clear_grid_cache(self):
        self._grid_cache.clear()

    def _normalize_heatmap(self, heatmap: torch.Tensor) -> torch.Tensor:
        bs, npoints, h, w = heatmap.shape
        heatmap = heatmap.to(dtype=self._work_dtype(heatmap))

        if self.softmax_normalized:
            return torch.softmax(
                heatmap.reshape(bs, npoints, -1),
                dim=-1,
            ).reshape(bs, npoints, h, w)

        heatmap_sum = torch.clamp(heatmap.sum([2, 3]), min=1e-6)
        return heatmap / heatmap_sum.view(bs, npoints, 1, 1)

    def weighted_mean(self, heatmap):
        batch, npoints, h, w = heatmap.shape

        yy, xx = self._make_grid(
            h,
            w,
            device=heatmap.device,
            dtype=heatmap.dtype,
        )
        yy = yy.view(1, 1, h, w)
        xx = xx.view(1, 1, h, w)

        yy_coord = (yy * heatmap).sum([2, 3])
        xx_coord = (xx * heatmap).sum([2, 3])
        return torch.stack([xx_coord, yy_coord], dim=-1)

    def unbiased_weighted_covariance(self, htp, means, num_dim_image=2, EPSILON=None):
        if EPSILON is None:
            EPSILON = self.EPSILON

        htp = htp.to(dtype=self._work_dtype(htp))
        means = means.to(device=htp.device, dtype=htp.dtype)

        batch_size, num_points, height, width = htp.shape
        yv, xv = self._make_grid(
            height,
            width,
            device=htp.device,
            dtype=htp.dtype,
        )

        xmean = means[:, :, 0]
        ymean = means[:, :, 1]

        xv_minus_mean = xv.view(1, 1, height, width) - xmean.view(
            batch_size, num_points, 1, 1
        )
        yv_minus_mean = yv.view(1, 1, height, width) - ymean.view(
            batch_size, num_points, 1, 1
        )

        vec = torch.stack((xv_minus_mean, yv_minus_mean), dim=2)
        vec = vec.reshape(batch_size * num_points, num_dim_image, height * width)

        weights = htp.reshape(batch_size * num_points, 1, height * width)
        covariance = torch.bmm(weights * vec, vec.transpose(1, 2))
        covariance = covariance.view(
            batch_size,
            num_points,
            num_dim_image,
            num_dim_image,
        )

        V_1 = htp.sum([2, 3]) + EPSILON
        V_2 = torch.pow(htp, 2).sum([2, 3]) + EPSILON
        denominator = (V_1 - (V_2 / V_1)).clamp_min(EPSILON)

        covariance = covariance / denominator.view(batch_size, num_points, 1, 1)

        # eigh expects symmetric matrices; this also reduces small AMP/kernel
        # asymmetries before decomposition.
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        return covariance

    def _maybe_check_finite(self, covars_flat: torch.Tensor) -> None:
        """Optional NaN/Inf guard on the covariance prior to eigh.

        Disabled by default: reading ``torch.isfinite(...).all()`` in a Python
        ``if`` forces a CUDA host-device sync every step. When
        ``check_finite_interval > 0`` the guard runs only every N calls so long
        runs keep protection without paying the sync each forward.
        """

        self._eigh_step += 1
        if not self.check_finite:
            return
        interval = self.check_finite_interval
        if interval > 0 and (self._eigh_step % interval) != 0:
            return
        if not torch.isfinite(covars_flat).all().item():
            raise ValueError(
                "STARLoss_v2 covariance contains NaN/Inf before eigendecomposition"
            )

    def covariance_eigh(
        self, covars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eigen-decompose 2x2 covariance matrices on tensor device.

        CUDA tensors stay CUDA. CPU fallback is only for unusual backends/builds
        where torch.linalg.eigh fails.
        """

        batch_size, num_points = covars.shape[:2]
        covars_flat = covars.reshape(batch_size * num_points, 2, 2)
        covars_flat = 0.5 * (covars_flat + covars_flat.transpose(-1, -2))

        self._maybe_check_finite(covars_flat)

        try:
            evalues, evectors = torch.linalg.eigh(covars_flat, UPLO="U")
        except RuntimeError:
            evalues, evectors = torch.linalg.eigh(covars_flat.cpu(), UPLO="U")
            evalues = evalues.to(covars_flat.device)
            evectors = evectors.to(covars_flat.device)

        evalues = evalues.view(batch_size, num_points, 2).to(
            device=covars.device,
            dtype=covars.dtype,
        )
        evectors = evectors.view(batch_size, num_points, 2, 2).to(
            device=covars.device,
            dtype=covars.dtype,
        )
        evalues = evalues.clamp_min(float(self.EPSILON))
        return evalues, evectors

    def ambiguity_guided_decompose(self, error, evalues, evectors):
        bs, npoints = error.shape[:2]
        error = error.to(device=evalues.device, dtype=evalues.dtype)

        normal_vector = evectors[:, :, 0]
        tangent_vector = evectors[:, :, 1]

        normal_error = torch.matmul(normal_vector.unsqueeze(-2), error.unsqueeze(-1))
        tangent_error = torch.matmul(tangent_vector.unsqueeze(-2), error.unsqueeze(-1))

        normal_error = normal_error.squeeze(dim=-1)
        tangent_error = tangent_error.squeeze(dim=-1)

        normal_dist = self.dist_func(
            normal_error,
            torch.zeros_like(normal_error),
            reduction="none",
        )
        tangent_dist = self.dist_func(
            tangent_error,
            torch.zeros_like(tangent_error),
            reduction="none",
        )

        normal_dist = normal_dist.reshape(bs, npoints, 1)
        tangent_dist = tangent_dist.reshape(bs, npoints, 1)
        dist = torch.cat((normal_dist, tangent_dist), dim=-1)

        scale_dist = dist / torch.sqrt(evalues + self.EPSILON)
        return scale_dist.sum(-1)

    def eigenvalue_restriction(self, evalues, batch, npoints):
        return torch.abs(evalues.view(batch, npoints, 2)).sum(-1)

    def per_point_loss_from_normalized_heatmap(self, heatmap, groundtruth):
        """Return STAR per-point loss from an already normalized heatmap.

        This is the safe reuse path: callers should only use it when `heatmap`
        is exactly the normalized probability map STARLoss_v2 would otherwise
        compute internally.
        """

        bs, npoints, h, w = heatmap.shape
        heatmap = heatmap.to(dtype=self._work_dtype(heatmap))
        groundtruth = groundtruth.to(device=heatmap.device, dtype=heatmap.dtype)

        means = self.weighted_mean(heatmap)
        covars = self.unbiased_weighted_covariance(heatmap, means)
        evalues, evectors = self.covariance_eigh(covars)

        loss_trans = self.ambiguity_guided_decompose(
            groundtruth - means,
            evalues,
            evectors,
        )
        loss_eigen = self.eigenvalue_restriction(evalues, bs, npoints)
        return loss_trans + self.w * loss_eigen

    def per_point_loss(self, heatmap, groundtruth, *, normalized_heatmap=False):
        """Return STARLoss_v2 per landmark with shape [B, N]."""

        if normalized_heatmap:
            return self.per_point_loss_from_normalized_heatmap(heatmap, groundtruth)

        return self.per_point_loss_from_normalized_heatmap(
            self._normalize_heatmap(heatmap),
            groundtruth,
        )

    def forward(self, heatmap, groundtruth):
        """
        heatmap:     B x N x H x W
        groundtruth: B x N x 2
        output:      scalar mean STAR loss
        """

        return self.per_point_loss(heatmap, groundtruth).mean()
