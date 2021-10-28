import math
import psutil
import argparse

import torch
from torch import tensor
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, AdamW, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification

from doc_simp.models.data.bert import BertDataModule


INPUTS = {
    "roberta": ["input_ids", "attention_mask", "labels"],
    "bert": ["input_ids", "attention_mask", "token_type_ids", "labels"]
}


def run_classifier(model, test_set, input_col="complex", max_samples=None, device="cuda", batch_size=16):
    if max_samples is not None:
        test_set = test_set[:max_samples]

    with torch.no_grad():
        # preprocess data
        dm = BertDataModule(model.tokenizer, hparams=model.hparams)
        test = dm.preprocess(list(test_set[input_col]))

        # prepare data loader
        features = INPUTS[model.model_type][:-1] # ignore labels
        dataset = TensorDataset(*[test[f].to(device) for f in features])
        test_data = DataLoader(dataset, batch_size=batch_size)

        # run predictions for each batch
        preds = []
        for batch in test_data:
            _batch = {features[i]:batch[i] for i in range(len(features))}
            output = model.model(**_batch, return_dict=True)
            _, logits = extract_results(output)
            preds += logits
    
    return preds

def extract_results(output):
    if type(output) is tuple:
        loss = output[0]
        logits = output[1]
    else:
        loss = None
        if "loss" in output:
            loss = output["loss"]
        logits = output["logits"]

    return loss, logits


class LightningBert(pl.LightningModule):

    def __init__(self, hparams, model_type=None, pt_model=None, num_labels=4):
        super().__init__()

        # resolve model type and pretrained base
        if model_type is None:
            model_type = hparams.model_type if hparams.model_type is not None else "roberta"
        if pt_model is None:
            pt_model = "roberta-base" if model_type == "roberta" else "bert-base-uncased"
        self.model_type = model_type
        self.pt_model = pt_model

        # set up model
        if model_type == "bert":
            self.model = BertForSequenceClassification.from_pretrained(pt_model, num_labels=num_labels)
            self.tokenizer = BertTokenizer.from_pretrained(pt_model, do_lower_case=True)
        elif model_type == "roberta":
            self.model = RobertaForSequenceClassification.from_pretrained(pt_model, num_labels=num_labels)
            self.tokenizer = RobertaTokenizer.from_pretrained(pt_model)
        else:
            raise ValueError("Unknown model type specified. Please choose one of [`bert`, `roberta`].")

        # basic hyperparams
        self.hparams = hparams
        self.learning_rate = self.hparams.learning_rate
        self.use_lr_scheduler = self.hparams.lr_scheduler
        self.train_check_interval = self.hparams.train_check_interval
        self.sys_log_interval = self.hparams.sys_log_interval

        self.num_labels = num_labels
        if "log_class_acc" not in self.hparams:
            self.log_class_acc = False
        else:
            self.log_class_acc = self.hparams.log_class_acc

        self.train_losses = []

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def training_step(self, batch, batch_idx):
        _batch = {INPUTS[self.model_type][i] : batch[i] for i in range(len(batch))}
        output = self.model(**_batch, return_dict=True)

        loss, _ = extract_results(output)

        output = {"loss": loss}

        # wandb logging
        self.train_losses.append(loss)
        train_inter = math.ceil(self.train_check_interval * self.trainer.num_training_batches)
        if batch_idx % train_inter == 0:
            avg_loss = torch.stack(self.train_losses).mean()
            self.logger.experiment.log({
                'train_loss': avg_loss
            })
            self.train_losses = []
        sys_inter = math.ceil(self.sys_log_interval * self.trainer.num_training_batches)
        if sys_inter > 0 and batch_idx % sys_inter == 0:
            self.logger.experiment.log({
                'cpu_memory_use': psutil.virtual_memory().percent
            })

        return output

    def validation_step(self, batch, batch_idx):
        _batch = {INPUTS[self.model_type][i] : batch[i] for i in range(len(batch))}
        output = self.model(**_batch, return_dict=True)
        loss, logits = extract_results(output)

        output = {
            "loss": loss,
            "preds": logits,
        }

        # accumalte relative acc for each class
        if self.log_class_acc:
            accs = [tensor([]) for _ in range(self.num_labels)]
            for i in range(len(logits)):
                ref = _batch["labels"][i]
                pred = logits[i].argmax()
                accs[ref] = torch.cat((accs[ref], tensor([pred == ref])))
            output["accs"] = accs

        return output

    def validation_epoch_end(self, outputs, prefix="val"):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        result = {
            f"{prefix}_loss": loss,
        }
        
        # log relative performance for each class
        if self.log_class_acc:
            for i in range(self.num_labels):
                agg = torch.stack([torch.mean(x["accs"][i]) for x in outputs]).mean()
                result[f"{prefix}_{i}_acc"] = agg

        # wandb log
        self.logger.experiment.log(result)

        return result


    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.01
                    },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.0
                    },
                ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        # use a learning rate scheduler if specified
        if self.use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }

        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument("--name", type=str, default=None, required=False,)
        parser.add_argument("--project", type=str, default=None, required=False,)
        parser.add_argument("--save_dir", type=str, default=None, required=False,)
        parser.add_argument("--checkpoint", type=str, default=None, required=False,)
        parser.add_argument("--wandb_id", type=str, default=None, required=False,)
        parser.add_argument("--model_type", type=str, default="roberta", required=False,)
        parser.add_argument("--x_col", type=str, default="x", required=False,)
        parser.add_argument("--y_col", type=str, default="y", required=False,)
        parser.add_argument("--lazy_loading", action="store_true", default=False)
        parser.add_argument("--train_check_interval", type=float, default=0.20)
        parser.add_argument("--sys_log_interval", type=float, default=0.0)
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--lr_scheduler", action="store_true")
        parser.add_argument("--log_class_acc", action="store_true", default=False)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--train_workers", type=int, default=8)
        parser.add_argument("--data_file", type=str, default=None, required=True)
        parser.add_argument("--data_file2", type=str, default=None, required=False)
        parser.add_argument("--max_samples", type=int, default=-1)
        parser.add_argument("--train_split", type=float, default=0.9)
        parser.add_argument("--val_split", type=float, default=0.05)
        parser.add_argument("--val_file", type=str, default=None, required=False)

        return parser
