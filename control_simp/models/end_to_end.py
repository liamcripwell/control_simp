import time
import psutil
import argparse
from typing import Dict, List
from collections import defaultdict

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from transformers import BartTokenizer, BartForConditionalGeneration

import control_simp.data.bart
from control_simp.utils import freeze_params, freeze_embeds, lmap, calculate_bleu


CONTROL_TOKENS = ["<ident>", "<para>", "<ssplit>", "<dsplit>"]
MTL_TOKENS = ["<clf>", "<gen>"]


def prepare_loader(dm, xx, yy=None, device="cuda", batch_size=16, num_workers=8):
    prep = dm.preprocess(xx, yy)

    dataset = TensorDataset(
        prep['input_ids'].to(device),
        prep['attention_mask'].to(device),
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    return loader


def run_generator(model, test_set, x_col="complex", ctrl_toks=None, max_samples=None, device="cuda", batch_size=16, num_workers=8, op_blacklist=[]):
    if max_samples is not None:
        test_set = test_set[:max_samples]

    with torch.no_grad():
        input_seqs = test_set if isinstance(test_set, list) else test_set[x_col].tolist()

        if ctrl_toks is not None:
            hold_outs = []
            new_seqs = []
            labels = ctrl_toks if isinstance(ctrl_toks, list) else test_set[ctrl_toks].tolist()
            for i in range(len(test_set)):
                if labels[i] in op_blacklist:
                    # hold out inputs of certain operation type
                    hold_outs.append(input_seqs[i])
                else:
                    hold_outs.append(None)
                    # prepend control token to input sequence
                    new_seqs.append(CONTROL_TOKENS[labels[i]] + " " + input_seqs[i])
            input_seqs = new_seqs

        # preprocess data
        dm = control_simp.data.bart.BartDataModule(model.tokenizer, hparams=model.hparams)
        test_data = prepare_loader(dm, input_seqs, device=device, batch_size=batch_size, num_workers=num_workers)

        # predict output sequences
        pred_ys = []
        for batch in test_data:
            input_ids, attention_mask = batch
            generated_ids = model.model.generate(
                input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                decoder_start_token_id=model.decoder_start_token_id,
                num_beams=model.eval_beams,
                max_length=128,
            )
            results = model.ids_to_clean_text(generated_ids)
            
            pred_ys += results

        # recombine predictions with held-out inputs
        if op_blacklist != []:
            j = 0
            for i in range(len(hold_outs)):
                if hold_outs[i] is None:
                    hold_outs[i] = pred_ys[j]
                    j += 1
            pred_ys = hold_outs

    return pred_ys



class BartFinetuner(pl.LightningModule):

    loss_names = ["loss"]
    metric_names = ["bleu"]
    default_val_metric = "bleu"

    def __init__(self, hparams=None):
        super().__init__()

        # load pretained model
        bart_model = BartForConditionalGeneration.from_pretrained(
            "facebook/bart-base", return_dict=True)
        self.model = bart_model

        # load pretained tokenizer and add new control tokens to vocab
        tokenizer = BartTokenizer.from_pretrained(
            'facebook/bart-base', add_prefix_space=True)
        self.tokenizer = tokenizer
        self.add_new_tokens()

        # load default model args if no hparams specified
        if hparams is None:
            parser = argparse.ArgumentParser()
            parser = self.add_model_specific_args(parser)
            hparams, _ = parser.parse_known_args()
            
        # basic hyperparams
        if isinstance(hparams, dict):
            self.save_hyperparameters(hparams)
        else:
            self.save_hyperparameters(dict(vars(hparams)))
        self.learning_rate = self.hparams.learning_rate
        self.use_lr_scheduler = self.hparams.lr_scheduler
        self.decoder_start_token_id = None  # default to config (self.pad?)

        # evaluation hyperparams
        if self.hparams.eval_max_gen_length is not None:
            self.eval_max_length = self.hparams.eval_max_gen_length
        else:
            self.eval_max_length = self.model.config.max_length
        self.metrics = defaultdict(list)
        self.eval_beams = self.model.config.num_beams if self.hparams.eval_beams is None else self.hparams.eval_beams
        self.val_metric = self.default_val_metric if self.hparams.val_metric is None else self.hparams.val_metric
        self.skip_val_gen = self.hparams.skip_val_gen

        # freeze params if required
        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())
        if self.hparams.freeze_embeds:
            freeze_embeds(self.model)

        # training loss cache to log mean every n steps
        self.train_losses = []

    def add_new_tokens(self):
        self.tokenizer.add_tokens(CONTROL_TOKENS + MTL_TOKENS, special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def training_step(self, batch, batch_idx):
        loss_tensors = self._step(batch)
        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}

        self.train_losses.append(loss_tensors[0])

        # wandb log
        if batch_idx % int(self.hparams.train_check_interval * self.trainer.num_training_batches) == 0:
            avg_loss = torch.stack(self.train_losses).mean()
            self.logger.experiment.log({'train_loss': avg_loss})
            self.train_losses = []

        if self.hparams.sys_log_interval > 0:
            if batch_idx % int(self.hparams.sys_log_interval * self.trainer.num_training_batches) == 0:
                self.logger.experiment.log({
                    'cpu_memory_use': psutil.virtual_memory().percent
                })

        return {"loss": loss_tensors[0]} #, "log": logs}

    def _step(self, batch):
        input_ids, attention_mask, labels = batch

        # shift the decoder input tokens to the right
        decoder_input_ids = self.shift_tokens_right(labels, self.pad)

        # run model and get the logits
        outputs = self(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False)
        lm_logits = outputs["logits"]

        # compute loss
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad)
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))

        return (loss,)

    def validation_step(self, batch, batch_idx) -> Dict:
        if self.skip_val_gen:
            loss_tensors = self._step(batch)
            val_results = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        else:
            val_results = self._generative_step(batch)
        return val_results

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        # compile val loss/metrics and calculate aggregates
        losses = {k: torch.stack([x[k] for x in outputs]).mean()
                  for k in self.loss_names}
        loss = losses["loss"]
        result = {
            f"{prefix}_loss": loss,
        }

        if not self.skip_val_gen:
            # add generative metric summaries to losses
            generative_metrics = {k: np.array([x[k] for x in outputs]).mean() 
                                    for k in self.metric_names + ["gen_time", "gen_len"]}
            metric_val = (generative_metrics[self.val_metric]
                        if self.val_metric in generative_metrics else losses[self.val_metric])
            metric_tensor: torch.FloatTensor = torch.tensor(metric_val).type_as(loss)
            result[f"{prefix}_{self.val_metric}"] = metric_tensor

            generative_metrics.update({k: v.item() for k, v in losses.items()})
            losses.update(generative_metrics)
        
        # wandb log
        self.logger.experiment.log(result)
        self.log("val_loss", loss)

        # callback writes this to self.metrics_save_path
        all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        self.metrics[prefix].append(all_metrics)
        # result["log"] = all_metrics

        return result

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def _generative_step(self, batch):
        t0 = time.time()
        input_ids, attention_mask, labels = batch

        # generate sequences from batch input
        generated_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            decoder_start_token_id=self.decoder_start_token_id,
            num_beams=self.eval_beams,
            max_length=self.eval_max_length,
        )
        gen_time = (time.time() - t0) / input_ids.shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(labels)

        # compute loss
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}

        # calculate other metrics
        bleu: Dict = self.calc_generative_metrics(preds, target)
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(
            gen_time=gen_time,
            gen_len=summ_len,
            preds=preds,
            target=target,
            **bleu)

        return base_metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

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

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def shift_tokens_right(self, input_ids, pad_token_id):
        """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
        prev_output_tokens = input_ids.clone()
        index_of_eos = (input_ids.ne(
            pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
        prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
        prev_output_tokens[:, 1:] = input_ids[:, :-1]
        return prev_output_tokens

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)
        return lmap(str.strip, gen_text)

    def calc_generative_metrics(self, preds, target) -> dict:
        bleu = calculate_bleu(preds, target)
        return {**bleu}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument("--name", type=str, default=None, required=False,)
        parser.add_argument("--project", type=str, default=None, required=False,)
        parser.add_argument("--save_dir", type=str, default=None, required=False,)
        parser.add_argument("--checkpoint", type=str, default=None, required=False,)
        parser.add_argument("--wandb_id", type=str, default=None, required=False,)
        parser.add_argument("--x_col", type=str, default="x", required=False,)
        parser.add_argument("--y_col", type=str, default="y", required=False,)
        parser.add_argument("--train_check_interval", type=float, default=0.20)
        parser.add_argument("--sys_log_interval", type=float, default=0.0)
        parser.add_argument("--skip_val_gen", action="store_true")
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--lr_scheduler", action="store_true")
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--train_workers", type=int, default=8)
        parser.add_argument("--data_file", type=str, default=None, required=False)
        parser.add_argument("--max_samples", type=int, default=-1)
        parser.add_argument("--train_split", type=float, default=0.9)
        parser.add_argument("--val_split", type=float, default=0.05)
        parser.add_argument("--val_file", type=str, default=None, required=False)
        parser.add_argument("--max_source_length", type=int, default=128)
        parser.add_argument("--max_target_length", type=int, default=128)
        parser.add_argument("--eval_beams", type=int, default=None, required=False)
        parser.add_argument("--val_metric", type=str, default=None, required=False,
            choices=["bleu", "rouge2", "loss",None])
        parser.add_argument("--eval_max_gen_length", type=int, default=None)
        parser.add_argument("--train_data_dir", type=str, default=None, required=False,)
        parser.add_argument("--valid_data_dir", type=str, default=None, required=False,)
        parser.add_argument("--use_mtl_toks", action="store_true")
        parser.add_argument("--simp_only", action="store_true")

        return parser
