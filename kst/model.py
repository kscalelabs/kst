""" K-Scale Labs Speech Tokenizer"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from kst.lfq import LookupFreeQuantization
from kst.seanet import SEANetDecoder, SEANetEncoder


@dataclass(kw_only=True)
class KSTConfig:
    n_filters: int
    dimension: int
    semantic_dimension: int
    sample_rate: int
    n_q: int
    strides: list[int]

    bidirectional: bool
    dilation_base: int
    residual_kernel_size: int
    n_residual_layers: int
    lstm_layers: int
    activation: str
    codebook_size: int
    encoding: str = "lfq"

    num_mels: int = 80
    num_freq: int = 1025
    n_fft: int = 1024
    hop_size: int = 240
    win_size: int = 1024
    fmin: int = 0
    fmax: int = 8000
    fmax_for_loss: Optional[int] = None


class KSTTokenizer(nn.Module):
    def __init__(self, config: KSTConfig):
        """
        
        Args:
            config (KSTConfig): Model configuration.
        """
        super().__init__()
        self.encoder = SEANetEncoder(
            n_filters=config.n_filters, 
            dimension=config.dimension, 
            ratios=config.strides,
            lstm=config.lstm_layers,
            bidirectional=config.bidirectional,
            dilation_base=config.dilation_base,
            residual_kernel_size=config.residual_kernel_size,
            n_residual_layers=config.n_residual_layers,
            activation=config.activation
        )
        self.sample_rate = config.sample_rate
        self.n_q = config.n_q
        self.downsample_rate = np.prod(config.strides)
        if config.dimension != config.semantic_dimension:
            self.transform = nn.Linear(config.dimension, config.semantic_dimension)
        else:
            self.transform = nn.Identity()

        self.encoding = config.encoding

        self.quantizer = LookupFreeQuantization(
            dim=config.dimension,
            codebook_size=config.codebook_size,
            num_codebooks=1,
        )
        self.decoder = SEANetDecoder(
            n_filters=config.n_filters, 
            dimension=config.dimension, 
            ratios=config.strides,
            lstm=config.lstm_layers,
            bidirectional=False,
            dilation_base=config.dilation_base,
            residual_kernel_size=config.residual_kernel_size,
            n_residual_layers=config.n_residual_layers,
            activation=config.activation
        )
        
    @classmethod
    def load_from_checkpoint(
        cls, 
        config: KSTConfig, 
        ckpt_path: str
    ):
        """ Load model from checkpoint.

        Args:
            config (KSTConfig): Model configuration.
            ckpt_path (str): Path of model checkpoint.

        Returns:
            KSTTokenizer: KSTTokenizer model.
        """
        model = cls(config)
        params = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(params)
        
        return model
    
    def encode(
        self, 
        x: torch.tensor, 
        n_q: int=None, 
        st: int=None
    ) -> torch.tensor:
        """ Encode wavs into codes.

        Args:
            x (torch.tensor): Input wavs. Shape: (batch, channels, timesteps).
            n_q (int, optional): Number of quantizers in RVQ used to encode. The default is all layers.
            st (int, optional): Start quantizer index in RVQ. The default is 0.

        Returns:
            codes (torch.tensor): Output indices for each quantizer. Shape: (n_q, batch, timesteps)
        """
        e = self.encoder(x)
        if st is None:
            st = 0
        n_q = n_q if n_q else self.n_q
        e = rearrange(e, 'b d t -> b t d')
        codes = self.quantizer.encode(e)
        return codes

    def encode_codes(self, 
        x: torch.tensor, 
        n_q: int=None,
        st: int=None
    ) -> torch.tensor:
        """ Encode wavs into codes.

        Args:
            x (torch.tensor): Input wavs. Shape: (batch, channels, timesteps).
            n_q (int, optional): Number of quantizers in RVQ used to encode. The default is all layers.
            st (int, optional): Start quantizer index in RVQ. The default is 0.

        Returns:
            codes (torch.tensor): Output indices for each quantizer. Shape: (n_q, batch, timesteps)
        """
        e = self.encoder(x)
        if st is None:
            st = 0
        n_q = n_q if n_q else self.n_q
        e = rearrange(e, 'b d t -> b t d')
        codes = self.quantizer.encode_codes(e)

        return codes

    def decode(
        self, 
        codes: torch.tensor, 
        st: int=0
    ) -> torch.tensor:
        """ Decode codes into wavs.

        Args:
            codes (torch.tensor): Indices for each quantizer. Shape: (n_q, batch, timesteps).
            st (int, optional): Start quantizer index in RVQ. The default is 0.

        Returns:
            o (torch.tensor): Reconstruct wavs from codes. Shape: (batch, channels, timesteps)
        """
        quantized = self.quantizer.decode(codes)
        quantized = rearrange(quantized, 'b t d -> b d t')

        o = self.decoder(quantized)
        return o

    def decode_codes(
        self, 
        codes: torch.tensor, 
        st: int=0
    ) -> torch.tensor:
        """ Decode codes into wavs.

        Args:
            codes (torch.tensor): Indices for each quantizer. Shape: (n_q, batch, timesteps).
            st (int, optional): Start quantizer index in RVQ. The default is 0.

        Returns:
            o (torch.tensor): Reconstruct wavs from codes. Shape: (batch, channels, timesteps)
        """
        quantized = self.quantizer.decode_codes(codes)
        quantized = rearrange(quantized, 'b t d -> b d t')
        o = self.decoder(quantized)

        return o
