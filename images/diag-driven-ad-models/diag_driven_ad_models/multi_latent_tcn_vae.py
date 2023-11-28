from diag_driven_ad_models.tcn_modules import Decoder, VaeEncoder
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List
import copy


class MultiLatentTcnVAE(pl.LightningModule):
    def __init__(
        self,
        enc_tcn1_in_dims: List[int] = [154, 50, 40, 30, 20],
        enc_tcn1_out_dims: List[int] = [50, 40, 30, 20, 10],
        enc_tcn2_in_dims: List[int] = [10, 6, 5, 4, 3],
        enc_tcn2_out_dims: List[int] = [6, 5, 4, 3, 1],
        dec_tcn1_in_dims: List[int] = [1, 3, 5, 6, 8],
        dec_tcn1_out_dims: List[int] = [3, 5, 6, 8, 10],
        dec_tcn2_in_dims: List[int] = [10, 10, 15, 20, 40],
        dec_tcn2_out_dims: List[int] = [10, 15, 20, 40, 6],
        beta: float = 0,
        seq_len: int = 500,
        kernel_size: int = 15,
        component_output_dims: list = [3, 3],
        component_latent_dims: list = [100, 100],
        total_latent_dim: int = 200,
        lr: float = 1e-3,
        dropout: float = 0.5,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.beta = beta

        self.save_hyperparameters()
        self.lr = lr
        self.component_output_dims = component_output_dims
        self.component_latent_dims = component_latent_dims

        self.comp_encoder_ls = nn.ModuleList([])
        for in_dim, ld_comp in zip(component_output_dims, component_latent_dims):
            tcn_input_dims = copy.copy(enc_tcn1_in_dims)
            tcn_input_dims[0] = in_dim
            self.comp_encoder_ls.append(
                VaeEncoder(
                    tcn1_in_dims=tcn_input_dims,
                    tcn1_out_dims=enc_tcn1_out_dims,
                    tcn2_in_dims=enc_tcn2_in_dims,
                    tcn2_out_dims=enc_tcn2_out_dims,
                    kernel_size=kernel_size,
                    latent_dim=ld_comp,
                    seq_len=seq_len,
                    dropout=dropout,
                )
            )

        self.full_decoder = Decoder(
            tcn1_in_dims=dec_tcn1_in_dims,
            tcn1_out_dims=dec_tcn1_out_dims,
            tcn2_in_dims=dec_tcn2_in_dims,
            tcn2_out_dims=dec_tcn2_out_dims,
            kernel_size=kernel_size,
            latent_dim=total_latent_dim,
            tcn1_seq_len=int(0.5 * seq_len),
            tcn2_seq_len=seq_len,
            dropout=dropout,
        )

        self.comp_decoder_ls = nn.ModuleList([])
        for out_dim, ld_comp in zip(component_output_dims, component_latent_dims):
            tcn_output_dims = copy.copy(dec_tcn2_out_dims)
            tcn_output_dims[-1] = out_dim
            self.comp_decoder_ls.append(
                Decoder(
                    tcn1_in_dims=dec_tcn1_in_dims,
                    tcn1_out_dims=dec_tcn1_out_dims,
                    tcn2_in_dims=dec_tcn2_in_dims,
                    tcn2_out_dims=tcn_output_dims,
                    kernel_size=kernel_size,
                    latent_dim=ld_comp,
                    tcn1_seq_len=int(0.5 * seq_len),
                    tcn2_seq_len=seq_len,
                    dropout=dropout,
                )
            )

    def encode(self, x_ls: List[torch.Tensor]) -> List[torch.Tensor]:
        return [enco(x) for x, enco in zip(x_ls, self.comp_encoder_ls)]

    def decode_comps(self, z_ls: List[torch.Tensor]):
        return [deco(z) for z, deco in zip(z_ls, self.comp_decoder_ls)]

    def loss_function(self, x_hat_full, x_full, x_hat_ls, x_comp_ls, pzx_ls):
        # initiating pz here since we ran into
        # problems when we did it in the init

        # compute pz list for components
        pz_ls = [
            torch.distributions.MultivariateNormal(
                loc=torch.zeros(lat_dim).to(self.device),
                covariance_matrix=torch.eye(lat_dim).to(self.device),
            )
            for lat_dim in self.component_latent_dims
        ]

        kl_ls = [
            torch.distributions.kl.kl_divergence(pzx, pz)
            for pzx, pz in zip(pzx_ls, pz_ls)
        ]
        kl_batch = torch.mean(torch.cat(kl_ls))
        mse_ls = [nn.MSELoss()(x, x_hat) for x, x_hat, in zip(x_comp_ls, x_hat_ls)]
        mse_full = nn.MSELoss()(x_full, x_hat_full)
        mse_ls_sum = sum(mse_ls)

        loss = mse_ls_sum + mse_full + self.beta * kl_batch
        return loss, mse_ls_sum, kl_batch

    @staticmethod
    def sample_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        e = torch.randn_like(std)
        s = e * std + mu
        return s

    def predict(self, batch):
        x, x_ls = batch
        if torch.cuda.is_available():
            x = x.cuda()
            x_ls = [x.cuda() for x in x_ls]
            self.cuda()
        pzx_param_list = self.encode(x_ls)
        z_ls = [mu_z for (mu_z, _) in pzx_param_list]
        z_full = torch.cat(z_ls, dim=1)
        x_hat_ls = self.decode_comps(z_ls)
        x_hat_full = self.full_decoder(z_full)
        comp_mse_ls = [
            torch.mean(torch.mean((x - x_hat) ** 2, dim=2), dim=1)
            for x, x_hat in zip(x_ls, x_hat_ls)
        ]
        mse_full = torch.mean(torch.mean((x - x_hat_full) ** 2, dim=2), dim=1)

        return comp_mse_ls, mse_full

    def shared_eval(self, x: torch.Tensor, x_comp_ls: List[torch.Tensor]):
        pzx_param_list = self.encode(x_comp_ls)
        # generate a pxz_sigma matrix for every element in pzx_param_list
        pzx_sigma_list = []
        for (_, log_var_z), lat_dim in zip(pzx_param_list, self.component_latent_dims):
            pzx_sigma_list.append(
                torch.cat(
                    [
                        torch.diag(torch.exp(log_var_z[i, :]))
                        for i in range(log_var_z.shape[0])
                    ]
                ).view(-1, lat_dim, lat_dim)
            )

        # generate one pzx distribution object for every element in pzx list
        pzx_ls = [
            torch.distributions.MultivariateNormal(
                loc=mu_z, covariance_matrix=pzx_sigma
            )
            for (mu_z, _), pzx_sigma in zip(pzx_param_list, pzx_sigma_list)
        ]

        # generate latend space samples for component decoders
        z_ls = [
            self.sample_gaussian(mu=mu_z, logvar=log_var_z)
            for (mu_z, log_var_z) in pzx_param_list
        ]

        # generate full latend space for full decoder
        z_full = torch.cat(z_ls, dim=1)

        # generate component signals
        x_hat_ls = self.decode_comps(z_ls)

        # generate full x_hat
        x_hat_full = self.full_decoder(z_full)

        loss, recon_loss, kl_batch = self.loss_function(
            x_hat_full, x, x_hat_ls, x_comp_ls, pzx_ls
        )

        return z_full, x_hat_ls, loss, recon_loss, kl_batch

    def training_step(self, batch, batch_idx):
        x, x_comp_ls = batch
        _, _, loss, recon_loss, kl_batch = self.shared_eval(x, x_comp_ls)
        self.log("train_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kl", kl_batch)
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_comp_ls = batch
        _, _, loss, recon_loss, kl_batch = self.shared_eval(x, x_comp_ls)
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
