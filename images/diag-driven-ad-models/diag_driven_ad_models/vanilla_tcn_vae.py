from diag_driven_ad_models.tcn_modules import Decoder, VaeEncoder
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List
import numpy as np


class VanillaTcnVAE(pl.LightningModule):
    def __init__(
        self,
        enc_tcn1_in_dims: List[int] = [154, 50, 40, 30, 20],
        enc_tcn1_out_dims: List[int] = [50, 40, 30, 20, 10],
        enc_tcn2_in_dims: List[int] = [10, 6, 5, 4, 3],
        enc_tcn2_out_dims: List[int] = [6, 5, 4, 3, 1],
        dec_tcn1_in_dims: List[int] = [1, 3, 5, 6, 8],
        dec_tcn1_out_dims: List[int] = [3, 5, 6, 8, 10],
        dec_tcn2_in_dims: List[int] = [10, 10, 30, 40, 50],
        dec_tcn2_out_dims: List[int] = [10, 30, 40, 50, 154],
        beta: float = 0,
        latent_dim: int = 10,
        seq_len: int = 500,
        kernel_size: int = 15,
        component_output_dims: list = [3, 3],
        lr: float = 1e-3,
        dropout: float = 0.5,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.beta = beta
        self.latent_dim = latent_dim

        self.lr = lr
        self.component_output_dims = component_output_dims

        self.encoder = VaeEncoder(
            tcn1_in_dims=enc_tcn1_in_dims,
            tcn1_out_dims=enc_tcn1_out_dims,
            tcn2_in_dims=enc_tcn2_in_dims,
            tcn2_out_dims=enc_tcn2_out_dims,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
            seq_len=seq_len,
            dropout=dropout,
        )
        self.decoder = Decoder(
            tcn1_in_dims=dec_tcn1_in_dims,
            tcn1_out_dims=dec_tcn1_out_dims,
            tcn2_in_dims=dec_tcn2_in_dims,
            tcn2_out_dims=dec_tcn2_out_dims,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
            tcn1_seq_len=int(0.5 * seq_len),
            tcn2_seq_len=seq_len,
            dropout=dropout,
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.encode(x)

    def loss_function(self, x, x_hat, pzx):
        # initiating pz here since we ran into
        # problems when we did it in the init
        pz = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.latent_dim).to(self.device),
            covariance_matrix=torch.eye(self.latent_dim).to(self.device),
        )

        kl = torch.distributions.kl.kl_divergence(pzx, pz)
        kl_batch = torch.mean(kl)
        recon_loss = nn.MSELoss()(x, x_hat)
        loss = recon_loss + self.beta * kl_batch
        return loss, recon_loss, kl_batch

    @staticmethod
    def sample_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        e = torch.randn_like(std)
        s = e * std + mu
        return s

    def predict(self, batch):
        x, _ = batch
        if torch.cuda.is_available():
            x = x.cuda()
            self.cuda()
        mu_z, _ = self.encode(x)
        z = mu_z
        x_hat = self.decode(z)
        mse_per_sig = torch.mean((x - x_hat) ** 2, dim=2)
        mse_full = torch.mean(mse_per_sig, dim=1)
        number_comp_signals_cumsum = list(np.array(self.component_output_dims).cumsum())
        comp_mse_ls = []
        for start, end in zip(
            [0] + number_comp_signals_cumsum[:-1], number_comp_signals_cumsum
        ):
            comp_mse_ls.append(torch.mean(mse_per_sig[:, start:end], dim=1))

        else:
            return comp_mse_ls, mse_full

    def shared_eval(self, x: torch.Tensor):
        mu_z, log_var_z = self.encode(x)
        pzx_sigma = torch.cat(
            [torch.diag(torch.exp(log_var_z[i, :])) for i in range(log_var_z.shape[0])]
        ).view(-1, self.latent_dim, self.latent_dim)
        pzx = torch.distributions.MultivariateNormal(
            loc=mu_z, covariance_matrix=pzx_sigma
        )
        z = self.sample_gaussian(mu=mu_z, logvar=log_var_z)
        x_hat = self.decode(z)
        loss, recon_loss, kl_batch = self.loss_function(x, x_hat, pzx)

        return z, x_hat, loss, recon_loss, kl_batch

    def training_step(self, batch, batch_idx):
        x, _ = batch
        _, _, loss, recon_loss, kl_batch = self.shared_eval(x)
        self.log("train_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kl", kl_batch)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        _, _, loss, recon_loss, kl_batch = self.shared_eval(x)
        self.log("val_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kl", kl_batch)
        return loss

    def configure_optimizers(self):
        """Configure optimizers for pytorch lightning."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
        )
        # scheduler = ReduceLROnPlateau(
        #     optimizer,
        # )
        return optimizer
