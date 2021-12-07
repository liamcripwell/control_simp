import os

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

import control_simp.models.end_to_end
from control_simp.utils import TokenFilter
from control_simp.data.loading import LazyPreproDataset


def pretokenize(model, data, save_dir, x_col="complex", y_col="simple", max_samples=None, chunk_size=32, ctrl_toks=False):
    if max_samples is not None:
        data = data[:max_samples]

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    dm = BartDataModule(model.tokenizer, hparams=dict(model.hparams))

    # number of chunks needed
    chunk_count = int(len(data)/chunk_size)+1

    for _, chunk in enumerate(np.array_split(data, chunk_count)):
        xx = []
        for i, row in chunk.iterrows():
            # add control tokens to beginning of inputs
            seq = control_simp.models.end_to_end.CONTROL_TOKENS[row.label] + " " if ctrl_toks else ""
            seq += row[x_col]
            xx.append(seq)

        tokd = dm.preprocess(xx, list(chunk[y_col]), pad_to_max_length=False, return_tensors=None)
        i = 0
        for j, _ in chunk.iterrows():
            a = torch.tensor(tokd["input_ids"][i])
            b = torch.tensor(tokd["attention_mask"][i])
            x = torch.stack([a, b])
            y = torch.tensor(tokd["labels"][i])
            torch.save(x, f"{save_dir}/{j}_x.pt")
            torch.save(y, f"{save_dir}/{j}_y.pt")
            i += 1

def pad_collate(batch):
    if len(batch[0]) == 3:
        (xx, mm, yy) = zip(*batch)
    else:
        (xx, mm, yy, zz, ll) = zip(*batch)

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=1)
    mm_pad = pad_sequence(mm, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=1)
    if len(batch[0]) == 5:
        zz_pad = pad_sequence(zz, batch_first=True, padding_value=1)
        ll_pad = pad_sequence(ll, batch_first=True, padding_value=1)
        return xx_pad, mm_pad, yy_pad, zz_pad, ll_pad
    else:
        return xx_pad, mm_pad, yy_pad


class BartDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, hparams):
        super().__init__()
        self.tokenizer = tokenizer

        # set hyperparams
        self.save_hyperparameters(hparams)
        self.x_col = self.hparams.x_col
        self.y_col = self.hparams.y_col
        self.batch_size = self.hparams.batch_size
        self.data_file = self.hparams.data_file
        self.max_samples = self.hparams.max_samples  # defaults to no restriction
        self.train_split = self.hparams.train_split  # default split will be 90/5/5
        self.val_split = min(self.hparams.val_split, 1-self.train_split)
        self.val_file = self.hparams.val_file
        self.max_source_length = self.hparams.max_source_length
        self.max_target_length = self.hparams.max_target_length
        self.train_data_dir = self.hparams.train_data_dir
        self.valid_data_dir = self.hparams.valid_data_dir
        self.train_workers = self.hparams.train_workers
        self.collate_fn = pad_collate if self.train_data_dir is not None else None
        self.use_mtl_toks = False if "use_mtl_toks" not in self.hparams else self.hparams.use_mtl_toks
        self.use_multihead = False if "use_multihead" not in self.hparams else self.hparams.use_multihead
        self.simp_only = False if "simp_only" not in self.hparams else self.hparams.simp_only

    def prepare_data(self):
        # NOTE: shouldn't assign state in here
        pass

    def setup(self, stage):
        # read and prepare input data
        self.data = pd.read_csv(self.data_file)
        if self.simp_only:
            print("Skipping 0 labels...")
            print(f"Original training samples: {len(self.data)}")
            self.data = self.data[self.data.label != 0]
            print(f"Selected training samples: {len(self.data)}")
        self.data = self.data.sample(frac=1)[:min(self.max_samples, len(self.data))] # NOTE: this will actually exclude the last item
        if self.val_file is not None:
            print("Loading specific validation samples...")
            self.validate = pd.read_csv(self.val_file)
            if self.simp_only:
                print("Skipping 0 labels...")
                print(f"Original validation samples: {len(self.validate)}")
                self.validate = self.validate[self.validate.label != 0]
                print(f"Selected validation samples: {len(self.validate)}")
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

        self.build_datasets()

    def build_datasets(self):
        if self.train_data_dir is None:
            # preprocess datasets
            if self.use_multihead:
                self.train = self.build_tensor_dataset(self.preprocess(
                    list(self.train[self.x_col]), list(self.train[self.y_col]), list(self.train["label"])))
                self.validate = self.build_tensor_dataset(self.preprocess(
                    list(self.validate[self.x_col]), list(self.validate[self.y_col]), list(self.validate["label"])))
                self.test = self.build_tensor_dataset(self.preprocess(
                    list(self.test[self.x_col]), list(self.test[self.y_col]), list(self.test["label"])))
            else:
                self.train = self.build_tensor_dataset(self.preprocess(
                    list(self.train[self.x_col]), list(self.train[self.y_col])))
                self.validate = self.build_tensor_dataset(self.preprocess(
                    list(self.validate[self.x_col]), list(self.validate[self.y_col])))
                self.test = self.build_tensor_dataset(self.preprocess(
                    list(self.test[self.x_col]), list(self.test[self.y_col])))
        else:
            # get control token ids
            ctrl_tok_ids = None
            mtl_tok_ids = None
            if self.use_mtl_toks:
                ctrl_tok_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(control_simp.models.end_to_end.CONTROL_TOKENS))
                mtl_tok_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(control_simp.models.end_to_end.MTL_TOKENS))

            # only use label_col when necessary, i.e mtl or multihead
            label_col = "label" if any([self.use_mtl_toks, self.use_multihead]) else None

            # prepare lazy loading datasets for pre-tokenized data
            self.train = LazyPreproDataset(
                self.train, self.train_data_dir, label_col=label_col, ctrl_tok_ids=ctrl_tok_ids, mtl_tok_ids=mtl_tok_ids)
            if self.val_file is not None:
                self.validate = LazyPreproDataset(
                    self.validate, self.valid_data_dir, label_col=label_col, ctrl_tok_ids=ctrl_tok_ids, mtl_tok_ids=mtl_tok_ids)
            else:
                self.validate = LazyPreproDataset(
                    self.validate, self.train_data_dir, label_col=label_col, ctrl_tok_ids=ctrl_tok_ids, mtl_tok_ids=mtl_tok_ids)
            self.test = LazyPreproDataset(
                self.test, self.train_data_dir, label_col=label_col, ctrl_tok_ids=ctrl_tok_ids, mtl_tok_ids=mtl_tok_ids)

    def build_tensor_dataset(self, data):
        return TensorDataset(
            data['input_ids'],
            data['attention_mask'],
            data['labels']
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.train_workers, 
                            pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=1, 
                            pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=1, 
                            pin_memory=True, collate_fn=self.collate_fn)

    def preprocess(self, source_sequences, target_sequences=None, labels=None, pad_to_max_length=True, return_tensors="pt"):
        """Transforms data into tokenized input/output sequences."""
        source_sequences = TokenFilter(max_len=self.max_source_length, blacklist=["<SEP>"])(source_sequences)
        transformed_x = self.tokenizer(source_sequences, max_length=self.max_source_length,
                            padding=pad_to_max_length, truncation=True, 
                            return_tensors=return_tensors, add_prefix_space=True)
        
        result = {
            "input_ids": transformed_x['input_ids'],
            "attention_mask": transformed_x['attention_mask'],
        }

        # preprocess target sequences if provided
        if target_sequences is not None:
            target_sequences = TokenFilter(max_len=self.max_target_length, blacklist=["<SEP>"])(target_sequences)
            transformed_y = self.tokenizer(target_sequences, max_length=self.max_target_length,
                                padding=pad_to_max_length, truncation=True, 
                                return_tensors=return_tensors, add_prefix_space=True)
            result["labels"] = transformed_y['input_ids']

        if labels is not None:
            result["clf_labels"] = torch.tensor(labels)

        return result