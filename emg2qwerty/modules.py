# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
# new addition
from emg2qwerty.charset import charset

import math
import torch
import torch.nn.functional as F
import numpy as np
import scipy.interpolate
from torch import nn


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)

# New Transformer Class
class TransformerEMGModel(nn.Module):
  def __init__(self, input_size: int, num_heads: int, hidden_size: int,
    num_layers: int, num_classes: int):
    super().__init__()

    # 1D Conv Layer
    self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm1d(input_size)
    self.relu = nn.ReLU()

    # Multi-Head Attention Layer
    self.attn_layer = nn.TransformerEncoderLayer(
      d_model=hidden_size,
      nhead=num_heads,
      dim_feedforward=4*hidden_size,
      dropout=0.1,
      batch_first=False
    )
    self.attn = nn.TransformerEncoder(self.attn_layer, num_layers=num_layers)

    # Feedforward Networks
    self.feedforward1 = nn.Linear(input_size, hidden_size)
    self.norm1 = nn.LayerNorm(hidden_size)
    self.feedforward2 = nn.Linear(hidden_size, hidden_size)
    self.norm2 = nn.LayerNorm(hidden_size)

    self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    self.dropout = nn.Dropout(0.1)

    self.output_layer = nn.Linear(hidden_size, num_classes)
  
  def forward(self, x):
    #x = x.permute(0, 2, 1)
    #x = self.conv1d(x)
    #x = self.bn1(x)
    #x = self.relu(x)

    print(f"[DEBUG] Input shape before Transformer: {x.shape}")

    # if x.shape[-1] != 64:  # Expecting d_model=64
        # raise ValueError(f"Expected feature size 64, but got {x.shape[-1]} instead.")

    x = x.permute(2, 0, 1)
    x = self.attn(x)

    x = self.feedforward1(x)
    x = self.norm1(x)
    x = self.relu(x)

    x = self.feedforward2(x)
    x = self.norm2(x)
    x = self.relu(x)

    x = x.permute(1, 2, 0)
    x = self.global_avg_pool(x).squeeze(-1)

    x = self.dropout(x)
    x = self.output_layer(x)

    return F.log_softmax(x, dim=-1)

# Transformer-RNN Hybrid
class HybridTransformerRNN(nn.Module):
  def __init__(self, input_size: int, num_heads: int, hidden_size: int, 
    num_layers: int, num_classes: int, rnn_type: str = "LTSM"):
    super().__init__()

    if rnn_type == "LSTM":
      self.rnn = nn.LSTM(
          input_size, 
          hidden_size, 
          num_layers, 
          batch_first=True, 
          bidirectional=True, 
          dropout=0.4
      )
    elif rnn_type == "GRU":
      self.rnn = nn.GRU(
          input_size, 
          hidden_size, 
          num_layers, 
          batch_first=True, 
          bidirectional=True, 
          dropout=0.4
      )
    else:
      raise ValueError("Unsupported RNN type. Choose from ['LTSM', 'GRU']")
    
    self.attn_layer = nn.TransformerEncoderLayer(
      d_model= 2 * hidden_size,
      nhead=num_heads,
      dim_feedforward= 4 * hidden_size,
      dropout=0.1,
      batch_first=True
    )
    self.attn = nn.TransformerEncoder(self.attn_layer, num_layers=num_layers)
    self.feedforward1 = nn.Linear(2*hidden_size, hidden_size)
    self.norm1 = nn.LayerNorm(hidden_size)
    self.feedforward2 = nn.Linear(hidden_size, hidden_size)
    self.norm2 = nn.LayerNorm(hidden_size)

    self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
    self.dropout = nn.Dropout(0.2)
    self.output_layer = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    x, _ = self.rnn(x)
    x = self.attn(x)
    x = self.feedforward1(x)
    x = self.norm1(x)
    x = F.relu(x)

    x = self.feedforward2(x)
    x = self.norm2(x)
    x = F.relu(x)

    # x = x.permute(0, 2, 1)
    # x = self.global_avg_pool(x).squeeze(-1)

    x = self.dropout(x)
    x = self.output_layer(x)

    return F.log_softmax(x, dim=-1)


# New RNN Class
class RecurrentEncoder(nn.Module):
  def __init__(self, input_size: int, hidden_size: int, 
    num_layers: int, num_classes: int, rnn_type: str = "LTSM"):
    super().__init__()
    self.rnn_type = rnn_type
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    if rnn_type == "LSTM":
      self.rnn = nn.LSTM(
          input_size, 
          hidden_size, 
          num_layers, 
          batch_first=False, 
          bidirectional=True, 
          dropout=0.3
      )
    elif rnn_type == "GRU":
      self.rnn = nn.GRU(
          input_size, 
          hidden_size, 
          num_layers, 
          batch_first=False, 
          bidirectional=True, 
          dropout=0.3
      )
    else:
      raise ValueError("Unsupported RNN type. Choose from ['LTSM', 'GRU']")

    # self.layer_norm = nn.LayerNorm(2*hidden_size)
    self.fc = TDSFullyConnectedBlock(2 * hidden_size)
    self.output_layer = nn.Linear(2 * hidden_size, input_size)
    

  def forward(self, x):
    outputs, _ = self.rnn(x) # (T, N, 2*hidden_size)
    outputs = self.fc(outputs)
    outputs = self.output_layer(outputs) # (T, N, num_classes)
    return nn.functional.log_softmax(outputs, dim=-1)

# Positional Encoding Class
class RelativePositionalEncoding(nn.Module):
  def __init__(self, hidden_size, max_len=10000):
    super().__init__()
    self.hidden_size = hidden_size
    self.max_len = max_len
    self.create_pe(max_len)
    
  def create_pe(self, max_len):
    pe = torch.zeros(max_len, self.hidden_size)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, self.hidden_size, 2) * (-math.log(10000.0) / self.hidden_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    self.register_buffer("pe", pe)

  def forward(self, x):
    T, N, C, = x.shape
    if T > self.max_len:
      self.max_len = T
      self.create_pe(self.max_len)

    return x + self.pe[:x.shape[0], :].unsqueeze(1).to(x.device)

class ConvFeatureExtractor(nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=5, stride=1, padding=2)
    self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
    self.bn = nn.BatchNorm1d(hidden_size)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = x.permute(1, 2, 0) # (T, N, C) -> (N, C, T)
    x = self.conv1(x)
    x = self.bn(self.relu(x))
    x = self.conv2(x)
    x = self.bn(self.relu(x))
    return x.permute(2, 0, 1)

# New Transformer Class
class TransformerEncoder(nn.Module):
  def __init__(self, input_size: int, num_heads: int, 
    hidden_size: int, num_layers: int):
    super().__init__()
    self.embedding = nn.Linear(input_size, hidden_size)
    # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # self.conv_frontend = ConvFeatureExtractor(input_size, hidden_size)
    # self.pos_encoding = RelativePositionalEncoding(hidden_size)
    encoder_layer = nn.TransformerEncoderLayer(
      d_model = hidden_size,
      nhead=num_heads,
      dim_feedforward=4*hidden_size,
      dropout=0.3,
      batch_first=False
    )
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    # self.output_layer = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    x = self.embedding(x)
    #x = x.to(self.device)
    #x = self.conv_frontend(x)
    #x = self.pos_encoding(x)
    x = self.transformer(x)
    # x = self.output_layer(x)
    # return nn.functional.log_softmax(x, dim =-1)
    return x


# New Decoder Class
class TransformerDecoder(nn.Module):
  def __init__(self, input_size: int, num_heads: int, 
    hidden_size: int, num_layers: int, num_classes: int):
    super().__init__()
    self.embedding = nn.Linear(input_size, hidden_size)

    decoder_layer = nn.TransformerDecoderLayer(
      d_model=hidden_size,
      nhead=num_heads,
      dim_feedforward=4*hidden_size,
      dropout=0.1,
      batch_first=False
    )

    self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
    self.output_layer = nn.Linear(hidden_size, num_classes)



  def forward(self, x, memory):
    x = self.embedding(x)
    x = self.transformer_decoder(x, memory)
    x = self.output_layer(x)
    return nn.functional.log_softmax(x, dim=-1)

class TransformerModule(nn.Module):
  def __init__(self, input_size: int, num_heads: int, 
    hidden_size: int, num_layers: int, num_classes: int):
    super().__init__()
    self.encoder = TransformerEncoder(input_size, num_heads, hidden_size, num_layers)
    self.decoder = TransformerDecoder(input_size, num_heads, hidden_size, num_layers, num_classes)

  def forward(self, x):
    memory = self.encoder(x)
    x = self.decoder(x, memory)
    return x
