import torch
import pytest

from control_simp.data.loading import LazyPreproDataset
from control_simp.models.end_to_end import BartFinetuner


@pytest.fixture
def tokenizer():
    model = BartFinetuner()
    return model.tokenizer

@pytest.fixture
def control_tokens():
    return ["<ident>", "<para>", "<ssplit>", "<dsplit>"]


def test_control_token_insertion(tokenizer, control_tokens):
    ctrl_tok_ids = tokenizer.convert_tokens_to_ids(control_tokens)
    ds = LazyPreproDataset([], "", label_tok_ids=ctrl_tok_ids)
    x = torch.tensor([0, 1, 2, 3, 4, 5])
    label = 1
    y = ds.insert_control_tok(x, label)
    target = torch.tensor([0, ctrl_tok_ids[label], 1, 2, 3, 4, 5])

    assert torch.equal(y, target)