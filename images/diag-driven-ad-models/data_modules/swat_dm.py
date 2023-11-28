import numpy as np
import pandas as pd
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import random
from typing import Optional


class SwatDataset(Dataset):
    """
    A dataset class for handling and preprocessing data from the SWaT dataset.

    Args:
        cols: List of column names to be used in the dataset.
        symbols_dct: A dictionary mapping component names to their corresponding symbols.
        df: DataFrame containing the SWaT dataset.
        seq_len_x: Length of the sequence for each sample.
        num_samples: Number of random samples to draw from the dataset. Defaults to 10,000.
        seed: Seed for the random number generator. Defaults to 42.
        random_samples: Flag to determine whether to select samples randomly. Defaults to True.

    Attributes:
        df (pd.DataFrame): Processed DataFrame with the selected columns.
        sample_index_list (list): List of indices for generating samples.
        x (torch.Tensor): Tensor representation of the entire dataset.
        comp_ls (list[torch.Tensor]): List of tensors, each representing a component's data.
        length (int): Total number of samples in the dataset.
        seq_len_x (int): Sequence length for each sample.
    """

    def __init__(
        self,
        cols: list[str],
        symbols_dct: dict,
        df: pd.DataFrame,
        seq_len_x: int,
        num_samples: Optional[int] = 10000,
        seed: Optional[int] = 42,
        random_samples: Optional[bool] = True,
    ):
        random.seed(seed)
        self.df = df[cols]
        if random_samples:
            self.sample_index_list = random.sample(
                list(range(0, len(self.df.index) - seq_len_x)),
                num_samples,
            )
        else:
            self.sample_index_list = list(range(len(df) - seq_len_x))
        self.x = torch.from_numpy(self.df.values.astype(np.float32))
        self.comp_ls = [
            torch.from_numpy(self.df[symbols_dct[comp]].values.astype(np.float32))
            for comp in sorted(symbols_dct.keys())
        ]
        self.length = len(self.sample_index_list)
        self.seq_len_x = seq_len_x

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            index: Index of the desired sample.

        Returns: A tuple containing the input sequence and the corresponding component sequences.
        """
        idx = self.sample_index_list[index]
        return (
            self.x[idx : idx + self.seq_len_x, :].T,
            [comp[idx : idx + self.seq_len_x, :].T for comp in self.comp_ls],
        )


class SwatDataModule(pl.LightningDataModule):
    """
    A data module class for the SWaT dataset, compatible with PyTorch Lightning.

    Args:
        df_train (pd.DataFrame): DataFrame containing the training data.
        df_val (pd.DataFrame): DataFrame containing the validation data.
        cols (list[str]): List of column names to be used in the dataset.
        symbols_dct (dict): A dictionary mapping component names to their corresponding symbols.
        batch_size (int): Batch size for the DataLoader. Defaults to 32.
        seq_len (int): Length of the sequence for each sample. Defaults to 1000.
        dl_workers (int): Number of workers for the DataLoader. Defaults to 8.
        num_train_samples (int): Number of samples to use from the training dataset. Defaults to 10,000.
        num_val_samples (int): Number of samples to use from the validation dataset. Defaults to 1,000.
        seed (int): Seed for random number generation. Defaults to 42.

    Attributes:
        batch_size (int): Batch size for the DataLoader.
        seq_len_x (int): Sequence length for each sample.
        cols (list[str]): List of column names used in the dataset.
        num_workers (int): Number of workers for data loading.
        train_ds (SwatDataset): Dataset object for the training data.
        val_ds (SwatDataset): Dataset object for the validation data.
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        cols: list[str],
        symbols_dct: dict,
        batch_size: int = 32,
        seq_len: int = 1000,
        dl_workers: int = 8,
        num_train_samples: int = 10000,
        num_val_samples: int = 1000,
        seed: int = 42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len_x = seq_len
        self.cols = cols
        self.num_workers = dl_workers

        self.train_ds = SwatDataset(
            df=df_train,
            seq_len_x=seq_len,
            cols=cols,
            symbols_dct=symbols_dct,
            num_samples=num_train_samples,
            seed=seed,
        )
        self.val_ds = SwatDataset(
            df=df_val,
            seq_len_x=seq_len,
            cols=cols,
            symbols_dct=symbols_dct,
            num_samples=num_val_samples,
            seed=seed,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
