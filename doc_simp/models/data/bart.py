import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from doc_simp.utils import TokenFilter


class BartDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, hparams):
        super().__init__()
        self.tokenizer = tokenizer

        # set hyperparams
        self.hparams = hparams
        self.x_col = self.hparams.x_col
        self.y_col = self.hparams.y_col
        self.batch_size = self.hparams.batch_size
        self.data_file = self.hparams.data_file
        self.max_samples = self.hparams.max_samples  # defaults to no restriction
        self.val_file = self.hparams.val_file
        self.train_split = self.hparams.train_split  # default split will be 90/5/5
        self.val_split = min(self.hparams.val_split, 1-self.train_split)
        self.max_source_length = self.hparams.max_source_length
        self.max_target_length = self.hparams.max_target_length

    def prepare_data(self):
        # NOTE: shouldn't assign state in here
        pass

    def setup(self, stage):
        # read and prepare input data
        self.data = pd.read_csv(self.data_file)
        self.data = self.data.sample(frac=1)[:min(self.max_samples, len(self.data))] # NOTE: this will actually exclude the last item
        if self.val_file is not None:
            print("Loading specific validation samples...")
            self.validate = pd.read_csv(self.val_file)
        print("All data loaded.")

        # train, validation, test split
        if self.val_file is None:
            train_span = int(self.train_split * len(self.data))
            val_span = int((self.train_split + self.val_split) * len(self.data))
            self.train, self.validate, self.test = np.split(
                self.data, [train_span, val_span])
        else:
            self.train = self.data
            self.test = self.data[:12] # arbitrarily have 12 test samples as precaution

        # preprocess datasets
        self.train = self.preprocess(
            list(self.train[self.x_col]), 
            list(self.train[self.y_col]))
        self.validate = self.preprocess(
            list(self.validate[self.x_col]), 
            list(self.validate[self.y_col]))
        self.test = self.preprocess(
            list(self.test[self.x_col]), 
            list(self.test[self.y_col]))

    def train_dataloader(self):
        dataset = TensorDataset(
            self.train['input_ids'],
            self.train['attention_mask'],
            self.train['labels'])
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True)

    def val_dataloader(self):
        dataset = TensorDataset(
            self.validate['input_ids'],
            self.validate['attention_mask'],
            self.validate['labels'])
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=1, pin_memory=True)

    def test_dataloader(self):
        dataset = TensorDataset(
            self.test['input_ids'],
            self.test['attention_mask'],
            self.test['labels'])
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=1, pin_memory=True)

    def preprocess(self, source_sequences, target_sequences, pad_to_max_length=True, return_tensors="pt"):
        """Transforms data into tokenized input/output sequences."""
        source_sequences = TokenFilter(max_len=self.max_source_length, blacklist=["<SEP>"])(source_sequences)

        transformed_x = self.tokenizer(source_sequences, max_length=self.max_source_length,
                            padding=pad_to_max_length, truncation=True, 
                            return_tensors=return_tensors, add_prefix_space=True)
        transformed_y = self.tokenizer(target_sequences, max_length=self.max_target_length,
                            padding=pad_to_max_length, truncation=True, 
                            return_tensors=return_tensors, add_prefix_space=True)

        return {
            "input_ids": transformed_x['input_ids'],
            "attention_mask": transformed_x['attention_mask'],
            "labels": transformed_y['input_ids'],
        }