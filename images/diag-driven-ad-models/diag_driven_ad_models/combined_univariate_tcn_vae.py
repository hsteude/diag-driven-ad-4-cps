from diag_driven_ad_models.vanilla_tcn_vae import VanillaTcnVAE
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List
import numpy as np


class CombinedUnivariateTcnVae(pl.LightningModule):
    def __init__(
        self,
        enc_tcn1_in_dims: List[int] = [1, 50, 40, 30, 20],
        enc_tcn1_out_dims: List[int] = [50, 40, 30, 20, 10],
        enc_tcn2_in_dims: List[int] = [10, 6, 5, 4, 3],
        enc_tcn2_out_dims: List[int] = [6, 5, 4, 3, 1],
        dec_tcn1_in_dims: List[int] = [1, 3, 5, 6, 8],
        dec_tcn1_out_dims: List[int] = [3, 5, 6, 8, 10],
        dec_tcn2_in_dims: List[int] = [10, 10, 30, 40, 50],
        dec_tcn2_out_dims: List[int] = [10, 30, 40, 50, 1],
        beta: float = 0,
        latent_dim: int = 10,
        seq_len: int = 500,
        kernel_size: int = 15,
        component_output_dims: list = [3, 3],
        lr: float = 1e-3,
        dropout: float = 0.5,
        *args,
        **kwargs,
    ):
        super(CombinedUnivariateTcnVae, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_models = sum(component_output_dims)
        self.component_output_dims = component_output_dims

        self.vae_instances = nn.ModuleList(
            [
                VanillaTcnVAE(
                    enc_tcn1_in_dims=enc_tcn1_in_dims,
                    enc_tcn1_out_dims=enc_tcn1_out_dims,
                    enc_tcn2_in_dims=enc_tcn2_in_dims,
                    enc_tcn2_out_dims=enc_tcn2_out_dims,
                    dec_tcn1_in_dims=dec_tcn1_in_dims,
                    dec_tcn1_out_dims=dec_tcn1_out_dims,
                    dec_tcn2_in_dims=dec_tcn2_in_dims,
                    dec_tcn2_out_dims=dec_tcn2_out_dims,
                    beta=beta,
                    latent_dim=latent_dim,
                    seq_len=seq_len,
                    kernel_size=kernel_size,
                    component_output_dims=component_output_dims,
                    lr=lr,
                    dropout=dropout,
                    *args,
                    **kwargs,
                )
                for _ in range(self.num_models)
            ]
        )

    def forward(self, x):
        return [vae(x_i) for vae, x_i in zip(self.vae_instances, x)]

    def predict(self, batch):
        x, _ = batch
        if torch.cuda.is_available():
            x = x.cuda()
            self.cuda()
        x_univar_ls = [
            x[:, i, :].reshape((-1, 1, x.shape[2])) for i in range(self.num_models)
        ]
        pred_resuls_ls = [
            vae.predict((xi, _))
            for vae, xi in zip(self.vae_instances, x_univar_ls)
        ]

        mse_per_sig = torch.stack([pred[0][0] for pred in pred_resuls_ls]).T
        mse_full = torch.mean(mse_per_sig, dim=1)
        number_comp_signals_cumsum = list(np.array(self.component_output_dims).cumsum())
        comp_mse_ls = []
        for start, end in zip(
            [0] + number_comp_signals_cumsum[:-1], number_comp_signals_cumsum
        ):
            comp_mse_ls.append(torch.mean(mse_per_sig[:, start:end], dim=1))

        return comp_mse_ls, mse_full

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_univar_ls = [
            x[:, i, :].reshape((x.shape[0], -1, x.shape[2]))
            for i in range(self.num_models)
        ]
        loss_ls = [
            vae.shared_eval(xi)[2] for vae, xi in zip(self.vae_instances, x_univar_ls)
        ]
        loss = sum(loss_ls) / self.num_models
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_univar_ls = [
            x[:, i, :].reshape((x.shape[0], -1, x.shape[2]))
            for i in range(self.num_models)
        ]
        loss_ls = [
            vae.shared_eval(xi)[2] for vae, xi in zip(self.vae_instances, x_univar_ls)
        ]
        loss = sum(loss_ls) / self.num_models
        self.log("val_loss", loss)
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
