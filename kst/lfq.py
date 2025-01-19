""" Lookup-free quantization implementation. """
import math

import torch
from torch import Tensor, nn


class LookupFreeQuantization(nn.Module):
    """Lookup-free quantization module.

    As opposed to nearest-neighbor (or lookup-based) quantization, this method
    first projects the input vector to the codebook dimension, then applies a
    binary quantization, so that each codebook is either zero or one.
    Representing this value as a binary number gives the codebook index.

    During training, we actually convert the values to -1 and 1, then project
    back to the original dimension. We train as a straight-through estimator,
    with some additional loss terms to encourage diversity and commitment.

    Parameters:
        dim: The input dimension. If not specified, it will be inferred from
            the codebook size as ``log2(codebook_size)``.
        codebook_size: The size of the codebook. If not specified, it will
            be inferred from the input dimension as ``2 ^ dim``. Must be a
            power of 2.
        entropy_loss_weight: The weight of the entropy loss.
        commitment_loss_weight: The weight of the commitment loss.
        diversity_gamma: The diversity gamma parameter.
        num_codebooks: The number of codebooks to use.
        codebook_scale: The codebook scale to use.
    """

    __constants__ = ["dim", "codebook_dim", "codebook_size", "num_codebooks", "codebook_scale"]

    def __init__(
        self,
        *,
        dim: int | None = None,
        codebook_size: int | None = None,
        entropy_loss_weight: float = 0.1,
        commitment_loss_weight: float = 1.0,
        diversity_gamma: float = 2.5,
        num_codebooks: int = 1,
        codebook_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if dim is None and codebook_size is None:
            raise ValueError("Either `dim` or `codebook_size` must be specified for LFQ.")

        # Gets the default number of codebooks, or validates if provided.
        if codebook_size is None:
            assert dim is not None
            codebook_size = 2**dim
        elif not math.log2(codebook_size).is_integer():
            suggested = 2 ** math.ceil(math.log2(codebook_size))
            raise ValueError(f"Your codebook size must be a power of 2 (suggested {suggested})")

        # Gets the default input dimension, or validates if provided.
        codebook_dim = round(math.log2(codebook_size))
        codebook_dims = codebook_dim * num_codebooks
        if dim is None:
            assert codebook_size is not None
            dim = codebook_dims
        
        # Projects the input to the codebook dimension.
        self.project_in = nn.Linear(dim, codebook_dims) if dim != codebook_dims else nn.Identity()
        self.project_out = nn.Linear(codebook_dims, dim) if dim != codebook_dims else nn.Identity()

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.codebook_scale = codebook_scale

        # Stores loss weights.
        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight
        self.commitment_loss_weight = commitment_loss_weight

        # Default loss values to use during inference.
        self.register_buffer("mask", 2 ** torch.arange(codebook_dim - 1, -1, -1), persistent=False)
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

        # For converting indices to codes.
        all_codes = torch.arange(codebook_size)
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        codebook = self.bits_to_codes(bits)
        self.register_buffer("codebook", codebook, persistent=False)

    def reset_parameters(self) -> None:
        self.mask.data.copy_(2 ** torch.arange(self.codebook_dim - 1, -1, -1).to(self.mask))
        self.zero.zero_()

        all_codes = torch.arange(self.codebook_size)
        bits = ((all_codes[..., None].to(self.mask) & self.mask) != 0).float()
        codebook = self.bits_to_codes(bits)
        self.codebook.data.copy_(codebook.to(self.codebook))

    mask: Tensor
    zero: Tensor
    codebook: Tensor

    def bits_to_codes(self, bits: Tensor) -> Tensor:
        return bits * self.codebook_scale * 2 - self.codebook_scale

    @property
    def dtype(self) -> torch.dtype:
        return self.codebook.dtype

    def quantize(self, x: Tensor, flat: bool = False) -> Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected final dimension to be {self.dim}, but got input shape {x.shape}")
        x = self.project_in(x)
        if flat:
            ids = 2 ** torch.arange(self.num_codebooks * self.codebook_dim, device=x.device)
        else:
            ids = 2 ** torch.arange(self.codebook_dim, device=x.device)
            x = x.unflatten(-1, (self.num_codebooks, self.codebook_dim))
        x_pos = (x > 0).long()
        x_quant = (x_pos * ids).to(x_pos).sum(-1)
        return x_quant

    def encode(self, x: Tensor, inv_temperature: float = 1.0) -> tuple[Tensor, Tensor]:
        x = self.project_in(x)
        x = x.unflatten(-1, (self.num_codebooks, self.codebook_dim))
        codebook_value = torch.ones_like(x) * self.codebook_scale
        x_pos = x > 0
        quantized = torch.where(x_pos, codebook_value, -codebook_value)
        return quantized
        
    def decode(self, quantized: Tensor, inv_temperature: float = 1.0) -> Tensor:
        # Projects the quantized codebook back to the original dimension.
        quantized = quantized.flatten(-2)
        quantized = self.project_out(quantized)
        return quantized
    
    def forward(self, x: Tensor, inv_temperature: float = 1.0) -> tuple[Tensor, Tensor]:
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected final dimension to be {self.dim}, but got input shape {x.shape}")
        # Projects to codebook dimensions.
        x = self.project_in(x)
        x = x.unflatten(-1, (self.num_codebooks, self.codebook_dim))

        # Does binary quantization of the codebook.
        codebook_value = torch.ones_like(x) * self.codebook_scale
        x_pos = x > 0
        quantized = torch.where(x_pos, codebook_value, -codebook_value)
        x = quantized

        # Compute the unique codebook indices.
        indices = (x_pos.int() * self.mask.int()).sum(-1)

        # Computes codebook commitment losses.
        commit_loss = self.zero

        # Projects the quantized codebook back to the original dimension.
        x = x.flatten(-2)
        x = self.project_out(x)

        # Gets the losses dataclass.
        commitment_loss = commit_loss * self.commitment_loss_weight

        return x, indices, commitment_loss
