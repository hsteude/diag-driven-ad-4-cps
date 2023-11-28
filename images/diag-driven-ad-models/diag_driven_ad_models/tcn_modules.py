import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from typing import Tuple


class Conv1DResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int,
        padding: str,
        stride: int,
        dilation: int,
        dropout: float,
    ):
        super(Conv1DResidualBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                in_dim,
                out_dim,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.activattion = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                out_dim,
                out_dim,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.residual = nn.Conv1d(in_dim, out_dim, 1)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activattion(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.activattion(out)
        out = self.dropout2(out)
        res = self.residual(x)
        return out + res


class TCN(nn.Module):
    def __init__(self, in_dims: list, out_dims: list, kernel_size: int, dropout: float):
        super(TCN, self).__init__()
        layers = []
        for i in range(len(in_dims)):
            dilation = 2**i
            layers.append(
                Conv1DResidualBlock(
                    in_dim=in_dims[i],
                    kernel_size=kernel_size,
                    out_dim=out_dims[i],
                    padding="same",
                    dilation=dilation,
                    stride=1,
                    dropout=dropout,
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(
        self,
        tcn1_in_dims: list,
        tcn1_out_dims: list,
        tcn2_in_dims: list,
        tcn2_out_dims: list,
        kernel_size: int = 15,
        latent_dim: int = 3,
        fc_in_out_dim: int = 10,
        tcn1_seq_len: int = 50,
        tcn2_seq_len: int = 50,
        dropout: float = 0.5,
    ) -> None:
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        self.fc_in = nn.Linear(
            in_features=latent_dim,
            out_features=int(
                fc_in_out_dim * tcn1_in_dims[0],
            ),
        )

        # tcn1
        self.upsampler1 = torch.nn.Upsample(size=tcn1_seq_len, mode="nearest")
        self.tcn1 = TCN(
            in_dims=tcn1_in_dims,
            out_dims=tcn1_out_dims,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # tcn2
        self.upsampler2 = torch.nn.Upsample(size=tcn2_seq_len, mode="nearest")
        self.tcn2 = TCN(
            in_dims=tcn2_in_dims,
            out_dims=tcn2_out_dims,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(self, z):
        # fc in
        out = self.fc_in(z).reshape(-1, 1, self.fc_in.out_features)
        out = self.upsampler1(out)
        out = self.tcn1(out)
        out = self.upsampler2(out)
        out = self.tcn2(out)
        return out


class VaeEncoder(nn.Module):
    def __init__(
        self,
        tcn1_in_dims: list,
        tcn1_out_dims: list,
        tcn2_in_dims: list,
        tcn2_out_dims: list,
        kernel_size: int = 15,
        latent_dim: int = 10,
        seq_len: int = 1000,
        dropout: float = 0.5,
    ) -> None:
        super(VaeEncoder, self).__init__()

        # TCN1
        self.tcn1 = TCN(
            in_dims=tcn1_in_dims,
            out_dims=tcn1_out_dims,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.max_pool1 = nn.MaxPool1d(kernel_size=2)

        # TCN2
        self.tcn2 = TCN(
            in_dims=tcn2_in_dims,
            out_dims=tcn2_out_dims,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.max_pool2 = nn.MaxPool1d(kernel_size=2)

        self.fc_z_mu = nn.Linear(
            in_features=int(0.25 * seq_len * tcn2_out_dims[-1]),
            out_features=latent_dim,
        )
        self.fc_z_logvar = nn.Linear(
            in_features=int(0.25 * seq_len * tcn2_out_dims[-1]),
            out_features=latent_dim,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.tcn1(x)
        out = self.max_pool1(out)
        out = self.tcn2(out)
        out = self.max_pool2(out)
        out = out.flatten(start_dim=1)
        mu_z = self.fc_z_mu(out)
        log_var_z = self.fc_z_logvar(out)
        return mu_z, torch.nn.Sigmoid()(log_var_z)


class Encoder(nn.Module):
    def __init__(
        self,
        tcn1_in_dims: list,
        tcn1_out_dims: list,
        tcn2_in_dims: list,
        tcn2_out_dims: list,
        kernel_size: int = 15,
        latent_dim: int = 10,
        seq_len: int = 1000,
        dropout: float = 0.5,
    ) -> None:
        super(Encoder, self).__init__()

        # TCN1
        self.tcn1 = TCN(
            in_dims=tcn1_in_dims,
            out_dims=tcn1_out_dims,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.max_pool1 = nn.MaxPool1d(kernel_size=2)

        # TCN2
        self.tcn2 = TCN(
            in_dims=tcn2_in_dims,
            out_dims=tcn2_out_dims,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.max_pool2 = nn.MaxPool1d(kernel_size=2)

        self.fc_out = nn.Linear(
            in_features=int(0.25 * seq_len * tcn2_out_dims[-1]), out_features=latent_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.tcn1(x)
        out = self.max_pool1(out)
        out = self.tcn2(out)
        out = self.max_pool2(out)
        out = out.flatten(start_dim=1)
        out = self.fc_out(out)
        return out
