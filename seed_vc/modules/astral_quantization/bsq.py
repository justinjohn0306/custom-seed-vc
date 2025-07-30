# Copyright (C) 2025 Human Dataware Lab.
# Modified from original work by HDL members
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Original work Copyright (C) 2025 Plachtaa <https://github.com/Plachtaa>
# Original source: https://github.com/Plachtaa/seed-vc

"""Lookup Free Quantization.

Proposed in https://arxiv.org/abs/2310.05737.

In the simplest setup, each dimension is quantized into {-1, 1}.
An entropy penalty is used to encourage utilization.
"""

from collections import namedtuple
from contextlib import nullcontext
from functools import cache, partial
from math import ceil, log2
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import pack, rearrange, reduce, unpack
from torch import einsum, nn
from torch.amp import autocast
from torch.distributed import nn as dist_nn
from torch.nn import Module

# constants

Return = namedtuple("Return", ["quantized", "indices", "entropy_aux_loss"])

LossBreakdown = namedtuple("LossBreakdown", ["per_sample_entropy", "batch_entropy", "commitment"])

DEFAULT_STRAIGHT_THROUGH_ACTIVATION = nn.Identity()

# distributed helpers


@cache
def is_distributed() -> bool:
    """Check if distributed training is enabled.

    Returns:
        True if distributed and world size > 1.
    """
    return dist.is_initialized() and dist.get_world_size() > 1


def maybe_distributed_mean(t: torch.Tensor) -> torch.Tensor:
    """Compute distributed mean if in distributed mode.

    Args:
        t: Input tensor.

    Returns:
        Mean tensor across all processes.
    """
    if not is_distributed():
        return t

    dist_nn.all_reduce(t)
    t = t / dist.get_world_size()
    return t


# helper functions


def exists(v) -> bool:
    """Check if value exists (is not None).

    Args:
        v: Value to check.

    Returns:
        True if value is not None.
    """
    return v is not None


def identity(t: torch.Tensor) -> torch.Tensor:
    """Identity function.

    Args:
        t: Input tensor.

    Returns:
        Same tensor unchanged.
    """
    return t


def default(*args):
    """Return first non-None value, calling it if callable.

    Args:
        *args: Values to check.

    Returns:
        First non-None value.
    """
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None


def pack_one(t: torch.Tensor, pattern: str) -> Tuple[torch.Tensor, list]:
    """Pack single tensor using einops.

    Args:
        t: Tensor to pack.
        pattern: Pack pattern.

    Returns:
        Packed tensor and shape info.
    """
    return pack([t], pattern)


def unpack_one(t: torch.Tensor, ps: list, pattern: str) -> torch.Tensor:
    """Unpack single tensor using einops.

    Args:
        t: Tensor to unpack.
        ps: Shape info.
        pattern: Unpack pattern.

    Returns:
        Unpacked tensor.
    """
    return unpack(t, ps, pattern)[0]


def l2norm(t: torch.Tensor) -> torch.Tensor:
    """L2 normalize tensor along last dimension.

    Args:
        t: Input tensor.

    Returns:
        L2 normalized tensor.
    """
    return F.normalize(t, dim=-1)


# entropy


def log(t: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Safe logarithm with epsilon clipping.

    Args:
        t: Input tensor.
        eps: Minimum value before log.

    Returns:
        Log of tensor.
    """
    return t.clamp(min=eps).log()


def entropy(prob: torch.Tensor) -> torch.Tensor:
    """Compute entropy of probability distribution.

    Args:
        prob: Probability tensor.

    Returns:
        Entropy values.
    """
    return (-prob * log(prob)).sum(dim=-1)


# cosine sim linear


class CosineSimLinear(Module):
    """Linear layer using cosine similarity.

    Attributes:
        scale: Scaling factor for output.
        weight: Weight matrix.
    """

    def __init__(self, dim_in: int, dim_out: int, scale: float = 1.0) -> None:
        """Initialize cosine similarity linear layer.

        Args:
            dim_in: Input dimension.
            dim_out: Output dimension.
            scale: Output scaling factor.
        """
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(dim_in, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using cosine similarity.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=0)
        return (x @ w) * self.scale


def soft_entropy_loss(u: torch.Tensor, tau: float = 1.0, gamma: float = 1.0) -> torch.Tensor:
    """Compute the soft entropy loss for Binary Spherical Quantization (BSQ).

    Args:
        u: Input latent embeddings of shape (batch_size, L).
        tau: Temperature scaling factor.
        gamma: Weight for the second entropy term.

    Returns:
        Soft entropy loss.
    """
    # Binary quantization: Generate implicit codebook corners
    L = u.size(1)  # Dimensionality of codebook
    corners = torch.tensor([-1.0, 1.0], device=u.device) / (L**0.5)

    # Compute soft quantization probabilities for all dimensions
    # q_hat(c|u) for each dimension
    prob_matrix = torch.sigmoid(
        2 * tau * corners.unsqueeze(1) * u.unsqueeze(2),
    )  # Shape: (batch_size, L, 2)

    # Entropy of q_hat(c|u) (independent along each dimension)
    entropy_per_dim = -torch.sum(prob_matrix * prob_matrix.log(), dim=-1)  # Shape: (batch_size, L)
    entropy_term1 = entropy_per_dim.mean()

    # Expected probabilities for dataset entropy (approximation)
    expected_probs = prob_matrix.mean(dim=0)  # Mean across batch, shape: (L, 2)
    entropy_term2 = -torch.sum(expected_probs * expected_probs.log(), dim=-1).mean()

    # Final entropy loss
    loss = entropy_term1 - gamma * entropy_term2
    return loss


# class


class BinarySphericalQuantize(Module):
    """Binary Spherical Quantization module.

    Implements lookup-free quantization where each dimension is quantized to {-1, 1}.
    Uses entropy penalties to encourage codebook utilization.

    Attributes:
        codebook_size: Size of the codebook.
        dim: Feature dimension.
        codebook_dim: Codebook dimension (log2 of codebook size).
        num_codebooks: Number of codebooks.
        project_in: Input projection layer.
        project_out: Output projection layer.
        entropy_loss_weight: Weight for entropy loss.
        commitment_loss_weight: Weight for commitment loss.
        diversity_gamma: Diversity loss weight.
        codebook_scale: Scale factor for codebook.
        spherical: Whether to use spherical normalization.
        soft_entropy_loss: Whether to use soft entropy loss.
    """

    def __init__(
        self,
        *,
        dim: Optional[int] = None,
        codebook_size: Optional[int] = None,
        entropy_loss_weight: float = 0.1,
        commitment_loss_weight: float = 0.0,
        diversity_gamma: float = 1.0,
        straight_through_activation: nn.Module = DEFAULT_STRAIGHT_THROUGH_ACTIVATION,
        num_codebooks: int = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        # for residual LFQ, codebook scaled down by 2x at each layer
        codebook_scale: float = 1.0,
        # make less than 1. to only use a random fraction of the probs for per sample entropy
        frac_per_sample_entropy: float = 0.25,
        has_projections: Optional[bool] = None,
        projection_has_bias: bool = True,
        soft_clamp_input_value: Optional[float] = None,
        cosine_sim_project_in: bool = False,
        cosine_sim_project_in_scale: Optional[float] = None,
        channel_first: Optional[bool] = None,
        experimental_softplus_entropy_loss: bool = False,
        # how much to shift the loss before softplus
        entropy_loss_offset: float = 5.0,
        # from https://arxiv.org/abs/2406.07548
        spherical: bool = True,
        # will force the quantization step to be full precision
        force_quantization_f32: bool = True,
        enable_entropy_loss: bool = True,
        soft_entropy_loss: bool = True,
    ) -> None:
        """Initialize Binary Spherical Quantization.

        Args:
            dim: Feature dimension.
            codebook_size: Size of codebook (must be power of 2).
            entropy_loss_weight: Weight for entropy auxiliary loss.
            commitment_loss_weight: Weight for commitment loss.
            diversity_gamma: Weight for diversity loss.
            straight_through_activation: Activation for straight-through estimation.
            num_codebooks: Number of codebooks.
            keep_num_codebooks_dim: Whether to keep codebook dimension.
            codebook_scale: Scale factor for codebook values.
            frac_per_sample_entropy: Fraction of samples for entropy calculation.
            has_projections: Whether to use input/output projections.
            projection_has_bias: Whether projections have bias.
            soft_clamp_input_value: Value for soft clamping input.
            cosine_sim_project_in: Whether to use cosine similarity for input projection.
            cosine_sim_project_in_scale: Scale for cosine similarity projection.
            channel_first: Whether channels are first dimension.
            experimental_softplus_entropy_loss: Whether to use softplus on entropy loss.
            entropy_loss_offset: Offset for softplus entropy loss.
            spherical: Whether to use spherical normalization.
            force_quantization_f32: Whether to force float32 for quantization.
            enable_entropy_loss: Whether to enable entropy loss.
            soft_entropy_loss: Whether to use soft entropy loss.
        """
        super().__init__()

        # some assert validations

        assert exists(dim) or exists(codebook_size), (
            "either dim or codebook_size must be specified for LFQ"
        )
        assert not exists(codebook_size) or log2(codebook_size).is_integer(), (
            "your codebook size must be a power of 2 for lookup free quantization "
            f"(suggested {2 ** ceil(log2(codebook_size))})"
        )

        codebook_size = default(codebook_size, lambda: 2**dim)
        self.codebook_size = codebook_size

        codebook_dim = int(log2(codebook_size))
        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)

        has_projections = default(has_projections, dim != codebook_dims)

        if cosine_sim_project_in:
            cosine_sim_project_in = default(cosine_sim_project_in_scale, codebook_scale)
            project_in_klass = partial(CosineSimLinear, scale=cosine_sim_project_in)
        else:
            project_in_klass = partial(nn.Linear, bias=projection_has_bias)

        self.project_in = project_in_klass(dim, codebook_dims) if has_projections else nn.Identity()
        self.project_out = (
            nn.Linear(codebook_dims, dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )
        self.has_projections = has_projections

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # channel first

        self.channel_first = channel_first

        # straight through activation

        self.activation = straight_through_activation

        # whether to use BSQ (binary spherical quantization)

        self.spherical = spherical
        self.maybe_l2norm = (lambda t: l2norm(t) * self.codebook_scale) if spherical else identity

        # entropy aux loss related weights

        assert 0 < frac_per_sample_entropy <= 1.0
        self.frac_per_sample_entropy = frac_per_sample_entropy

        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

        # codebook scale

        self.codebook_scale = codebook_scale

        # commitment loss

        self.commitment_loss_weight = commitment_loss_weight

        # whether to soft clamp the input value from -value to value

        self.soft_clamp_input_value = soft_clamp_input_value
        assert not exists(soft_clamp_input_value) or soft_clamp_input_value >= codebook_scale

        # whether to make the entropy loss positive through a softplus
        # (experimental, please report if this worked or not in discussions)

        self.entropy_loss_offset = entropy_loss_offset
        self.experimental_softplus_entropy_loss = experimental_softplus_entropy_loss

        # for no auxiliary loss, during inference

        self.register_buffer("mask", 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

        # whether to force quantization step to be f32

        self.force_quantization_f32 = force_quantization_f32

        # codes
        self.enable_entropy_loss = enable_entropy_loss
        self.soft_entropy_loss = soft_entropy_loss
        if codebook_size <= 100000:
            all_codes = torch.arange(codebook_size)
            bits = ((all_codes[..., None].int() & self.mask) != 0).float()
            codebook = self.bits_to_codes(bits)

            self.register_buffer("codebook", codebook.float(), persistent=False)
        else:
            all_codes = torch.arange(pow(2, 16))
            mask = 2 ** torch.arange(16 - 1, -1, -1)
            bits = ((all_codes[..., None].int() & mask) != 0).float()
            codebook = self.bits_to_codes(bits)

            self.register_buffer("codebook", codebook.float(), persistent=False)

    def bits_to_codes(self, bits: torch.Tensor) -> torch.Tensor:
        """Convert bits to codes.

        Args:
            bits: Binary bits tensor.

        Returns:
            Codes tensor.
        """
        return bits * self.codebook_scale * 2 - self.codebook_scale

    @property
    def dtype(self) -> torch.dtype:
        """Get dtype of codebook.

        Returns:
            Data type of codebook.
        """
        return self.codebook.dtype

    def indices_to_codes(self, indices: torch.Tensor, project_out: bool = True) -> torch.Tensor:
        """Convert indices to codes.

        Args:
            indices: Codebook indices.
            project_out: Whether to project codes to output dimension.

        Returns:
            Decoded codes.
        """
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        should_transpose = default(self.channel_first, is_img_or_video)

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... -> ... 1")

        # indices to codes, which are bits of either -1 or 1

        bits = ((indices[..., None].int() & self.mask) != 0).to(self.dtype)

        codes = self.bits_to_codes(bits)

        codes = self.maybe_l2norm(codes)

        codes = rearrange(codes, "... c d -> ... (c d)")

        # whether to project codes out to original dimensions
        # if the input feature dimensions were not log2(codebook size)

        if project_out:
            codes = self.project_out(codes)

        # rearrange codes back to original shape

        if should_transpose:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    def bits_to_z(self, bits: torch.Tensor) -> torch.Tensor:
        """Convert bits to latent representation.

        Args:
            bits: Binary bits tensor (must contain only -1 and 1).

        Returns:
            Latent representation.
        """
        # assert bits must contain only -1 and 1
        assert torch.all(bits.abs() == 1)
        quantized = bits.float()
        quantized = self.maybe_l2norm(quantized)
        z = self.project_out(quantized)
        return z

    def forward(
        self,
        x: torch.Tensor,
        inv_temperature: float = 100.0,
        return_loss_breakdown: bool = False,
        mask: Optional[torch.Tensor] = None,
        return_bits: bool = False,
    ) -> Union[torch.Tensor, Return, Tuple[Return, LossBreakdown]]:
        """Forward pass of Binary Spherical Quantization.

        Einstein notation:
            b - batch
            n - sequence (or flattened spatial dimensions)
            d - feature dimension, which is also log2(codebook size)
            c - number of codebook dim

        Args:
            x: Input tensor.
            inv_temperature: Inverse temperature for entropy calculation.
            return_loss_breakdown: Whether to return detailed loss breakdown.
            mask: Optional mask for valid positions.
            return_bits: Whether to return quantized bits directly.

        Returns:
            Quantized output, indices, and auxiliary loss.
            If return_loss_breakdown is True, also returns loss breakdown.
        """
        is_img_or_video = x.ndim >= 4
        should_transpose = default(self.channel_first, is_img_or_video)

        # standardize image or video into (batch, seq, dimension)

        if should_transpose:
            x = rearrange(x, "b d ... -> b ... d")
            x, ps = pack_one(x, "b * d")

        assert x.shape[-1] == self.dim, (
            f"expected dimension of {self.dim} but received {x.shape[-1]}"
        )

        x = self.project_in(x)

        # maybe soft clamp

        if exists(self.soft_clamp_input_value):
            clamp_value = self.soft_clamp_input_value
            x = (x / clamp_value).tanh() * clamp_value

        # split out number of codebooks

        x = rearrange(x, "b n (c d) -> b n c d", c=self.num_codebooks)

        # maybe l2norm

        x = self.maybe_l2norm(x)

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32

        quantization_context = (
            partial(autocast, "cuda", enabled=False) if force_f32 else nullcontext
        )

        with quantization_context():
            if force_f32:
                orig_dtype = x.dtype
                x = x.float()

            # quantize by eq 3.

            original_input = x

            codebook_value = torch.ones_like(x) * self.codebook_scale
            quantized = torch.where(x > 0, codebook_value, -codebook_value)
            if return_bits:
                return quantized

            # calculate indices

            indices = reduce((quantized > 0).int() * self.mask.int(), "b n c d -> b n c", "sum")

            # maybe l2norm

            quantized = self.maybe_l2norm(quantized)

            # use straight-through gradients (optionally with custom activation fn) if training

            if self.training:
                x = self.activation(x)
                x = x + (quantized - x).detach()
            else:
                x = quantized

            # entropy aux loss
            if self.soft_entropy_loss:
                entropy_aux_loss = soft_entropy_loss(x, tau=1.0, gamma=1.0)
            elif self.training and self.enable_entropy_loss:
                if force_f32:
                    codebook = self.codebook.float()

                codebook = self.maybe_l2norm(codebook)

                # whether to only use a fraction of probs, for reducing memory

                if self.frac_per_sample_entropy < 1.0:
                    # account for mask
                    if exists(mask):
                        original_input = original_input[mask]
                    original_input = rearrange(original_input, "b n ... -> (b n) ...")

                    rand_mask = torch.randn(self.codebook_dim).argsort(dim=-1) < 16

                    sampled_input = original_input[..., rand_mask]

                    sampled_distance = -2 * einsum(
                        "... i d, j d -> ... i j",
                        sampled_input,
                        codebook,
                    )

                    sampled_prob = (-sampled_distance * inv_temperature).softmax(dim=-1)

                    per_sample_probs = sampled_prob
                else:
                    if exists(mask):
                        original_input = original_input[mask]
                    original_input = rearrange(original_input, "b n ... -> (b n) ...")
                    # the same as euclidean distance up to a constant
                    distance = -2 * einsum("... i d, j d -> ... i j", original_input, codebook)

                    prob = (-distance * inv_temperature).softmax(dim=-1)

                    per_sample_probs = prob

                # calculate per sample entropy

                per_sample_entropy = entropy(per_sample_probs).mean()

                # distribution over all available tokens in the batch

                avg_prob = reduce(per_sample_probs, "... c d -> c d", "mean")

                avg_prob = maybe_distributed_mean(avg_prob)

                codebook_entropy = entropy(avg_prob).mean()

                # 1. entropy will be nudged to be low for each code,
                #    to encourage the network to output confident predictions
                # 2. codebook entropy will be nudged to be high, to encourage
                #    all codes to be uniformly used within the batch

                entropy_aux_loss = per_sample_entropy - self.diversity_gamma * codebook_entropy
            else:
                # if not training, just return dummy 0
                entropy_aux_loss = per_sample_entropy = codebook_entropy = self.zero

            # whether to make the entropy loss positive or not through a (shifted) softplus

            if self.training and self.experimental_softplus_entropy_loss:
                entropy_aux_loss = F.softplus(entropy_aux_loss + self.entropy_loss_offset)

            # commit loss

            if self.training and self.commitment_loss_weight > 0.0:
                commit_loss = F.mse_loss(original_input, quantized.detach(), reduction="none")

                if exists(mask):
                    commit_loss = commit_loss[mask]

                commit_loss = commit_loss.mean()
            else:
                commit_loss = self.zero

            # input back to original dtype if needed

            if force_f32:
                x = x.type(orig_dtype)

        # merge back codebook dim

        x = rearrange(x, "b n c d -> b n (c d)")

        # project out to feature dimension if needed

        x = self.project_out(x)

        # reconstitute image or video dimensions

        if should_transpose:
            x = unpack_one(x, ps, "b * d")
            x = rearrange(x, "b ... d -> b d ...")

            indices = unpack_one(indices, ps, "b * c")

        # whether to remove single codebook dim

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        # complete aux loss

        aux_loss = (
            entropy_aux_loss * self.entropy_loss_weight + commit_loss * self.commitment_loss_weight
        )

        # returns

        ret = Return(x, indices, aux_loss)

        if not return_loss_breakdown:
            return ret

        return ret, LossBreakdown(per_sample_entropy, codebook_entropy, commit_loss)


class GroupedResidualBSQ(Module):
    """Grouped Residual Binary Spherical Quantization.

    Splits features into groups and applies BSQ to each group independently.

    Attributes:
        dim: Total feature dimension.
        groups: Number of groups.
        accept_image_fmap: Whether to accept image feature maps.
        rvqs: List of BSQ modules for each group.
        codebook_size: Size of codebook.
    """

    def __init__(
        self,
        *,
        dim: int,
        groups: int = 1,
        accept_image_fmap: bool = False,
        **kwargs,
    ) -> None:
        """Initialize grouped residual BSQ.

        Args:
            dim: Total feature dimension.
            groups: Number of groups to split features.
            accept_image_fmap: Whether to accept image feature maps.
            **kwargs: Additional arguments for BSQ modules.
        """
        super().__init__()
        self.dim = dim
        self.groups = groups
        assert (dim % groups) == 0
        dim_per_group = dim // groups

        self.accept_image_fmap = accept_image_fmap

        self.rvqs = nn.ModuleList([])

        for _ in range(groups):
            self.rvqs.append(LFQ(dim=dim_per_group, **kwargs))

        self.codebook_size = self.rvqs[0].codebook_size

    @property
    def codebooks(self) -> torch.Tensor:
        """Get stacked codebooks from all groups.

        Returns:
            Stacked codebooks tensor.
        """
        return torch.stack(tuple(rvq.codebooks for rvq in self.rvqs))

    @property
    def split_dim(self) -> int:
        """Get dimension to split along.

        Returns:
            Split dimension (1 for images, -1 otherwise).
        """
        return 1 if self.accept_image_fmap else -1

    def get_codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Get codes from indices for all groups.

        Args:
            indices: Indices for each group.

        Returns:
            Stacked codes.
        """
        codes = tuple(
            rvq.get_codes_from_indices(chunk_indices)
            for rvq, chunk_indices in zip(self.rvqs, indices, strict=False)
        )
        return torch.stack(codes)

    def get_output_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Get output from indices for all groups.

        Args:
            indices: Indices for each group.

        Returns:
            Concatenated outputs.
        """
        outputs = tuple(
            rvq.get_output_from_indices(chunk_indices)
            for rvq, chunk_indices in zip(self.rvqs, indices, strict=False)
        )
        return torch.cat(outputs, dim=self.split_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_all_codes: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through grouped BSQ.

        Args:
            x: Input tensor.
            return_all_codes: Whether to return all codes.

        Returns:
            Tuple of (quantized output, indices, auxiliary loss).
        """
        shape, split_dim = x.shape, self.split_dim
        assert shape[split_dim] == self.dim

        # split the feature dimension into groups

        x = x.chunk(self.groups, dim=split_dim)

        forward_kwargs = dict()

        # invoke residual vq on each group

        out = tuple(rvq(chunk, **forward_kwargs) for rvq, chunk in zip(self.rvqs, x, strict=False))
        out = tuple(zip(*out, strict=False))

        # otherwise, get all the zipped outputs and combine them

        quantized, all_indices, *maybe_aux_loss = out

        quantized = torch.cat(quantized, dim=split_dim)
        all_indices = torch.stack(all_indices)

        ret = (quantized, all_indices, *maybe_aux_loss)
        return ret
