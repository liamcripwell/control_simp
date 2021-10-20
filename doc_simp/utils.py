import itertools
from typing import Callable, Iterable, List

from torch import nn
from sacrebleu import corpus_bleu


def lmap(f: Callable, x: Iterable) -> List:
    return list(map(f, x))


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]

def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart."""
    freeze_params(model.model.shared)
    for d in [model.model.encoder, model.model.decoder]:
        freeze_params(d.embed_positions)
        freeze_params(d.embed_tokens)

def calculate_bleu(output_lns, refs_lns, **kwargs) -> dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    return {
        "bleu": round(
            corpus_bleu(
                output_lns,
                [refs_lns],
                **kwargs).score,
            4)}

class TokenFilter():

    def __init__(self, max_len=-1, blacklist=[]) -> None:
        self.max_len = max_len
        self.blacklist = blacklist

    def __call__(self, seqs) -> str:
        if not isinstance(seqs, List): seqs = [seqs]

        for i in range(len(seqs)):
            buff = []
            toks = str(seqs[i]).split()
            for j in range(len(toks)):
                if toks[j] not in self.blacklist:
                    buff.append(toks[j])
                if len(buff) == self.max_len: break
            seqs[i] = " ".join(buff)

        return seqs

