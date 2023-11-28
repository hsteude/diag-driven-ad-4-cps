import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, List
import random
import torch
from loguru import logger


class SimDataSet(Dataset):
    """
    A custom dataset class for handling simulated data in PyTorch.

    Args:
        dataframe: The DataFrame containing the simulation data.
        subsystems_map: A mapping of subsystem names to their
                                         corresponding column names in the DataFrame.
        input_cols: List of column names to be used as input features.
        number_of_samples: Number of samples to be generated from the DataFrame.
        seq_len: Length of the sequence for each sample.
        seed: Seed for random number generation. Defaults to 42.

    Attributes:
        df (pd.DataFrame): DataFrame containing the simulation data.
        seq_len (int): Sequence length for each sample.
        length (int): Number of samples in the dataset.
        input_cols (List): List of column names used as input features.
        subsystems_map (Dict[str, List]): Mapping of subsystems to their columns.
        start_idx_ls (List): List of starting indices for each sample in the dataset.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        subsystems_map: Dict[str, List],
        input_cols: List,
        number_of_samples: int,
        seq_len: int,
        seed: int = 42,
    ):
        self.df = dataframe
        self.seq_len = seq_len
        self.length = number_of_samples
        self.input_cols = input_cols
        self.subsystems_map = subsystems_map
        start_idx_ls = list(range(len(dataframe) - seq_len))
        random.seed(seed)
        logger.info(f"Random seed in Dataset set to {seed}")
        self.start_idx_ls = random.sample(start_idx_ls, k=number_of_samples)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Retrieves a single item from the dataset.

        Args:
            index: Index of the item to retrieve.

        Returns: A tuple containing the input sequence and the corresponding
                   sequences for each subsystem.
        """
        idx = self.start_idx_ls[index]
        return (
            self.df[idx : idx + self.seq_len][self.input_cols]
            .values.astype(np.float32)
            .T,
            [
                self.df[idx : idx + self.seq_len][self.subsystems_map[subsys]]
                .values.astype(np.float32)
                .T
                for subsys in sorted(self.subsystems_map.keys())
            ],
        )


class SimDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        batch_size: int,
        input_cols: List[str],
        subsystems_map: Dict[str, List],
        number_of_train_samples: int,
        number_of_val_samples: int,
        seq_len: int,
        num_workers: int = 20,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ds_train = SimDataSet(
            dataframe=df_train,
            input_cols=input_cols,
            subsystems_map=subsystems_map,
            number_of_samples=number_of_train_samples,
            seq_len=seq_len,
            seed=seed,
        )
        self.ds_val = SimDataSet(
            dataframe=df_val,
            input_cols=input_cols,
            subsystems_map=subsystems_map,
            number_of_samples=number_of_val_samples,
            seq_len=seq_len,
            seed=seed,
        )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
