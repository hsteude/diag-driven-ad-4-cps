from joblib import load
import torch


class GMMWrapper:
    def __init__(
        self,
        gmm_model_a_path: str,
        gmm_model_b_path: str,
        gmm_model_full_path: str,
    ):
        self.gmm_a = load(gmm_model_a_path)
        self.gmm_b = load(gmm_model_b_path)
        self.gmm_full = load(gmm_model_full_path)

    def predict(self, batch):
        x, (x_a, x_b) = batch
        mse_a_ls = []
        mse_b_ls = []
        mse_full_ls = []
        for sample in range(x.shape[0]):
            mse_a_ls.append(-self.gmm_a.score_samples(x_a[sample, :, :].numpy().T).mean())
            mse_b_ls.append(-self.gmm_b.score_samples(x_b[sample, :, :].numpy().T).mean())
            mse_full_ls.append(-self.gmm_full.score_samples(x[sample, :, :].numpy().T).mean())

        return [torch.tensor(mse_a_ls), torch.tensor(mse_b_ls)], torch.tensor(mse_full_ls)
