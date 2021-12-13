import pytest

from control_simp.models.eval import calculate_split_acc

@pytest.fixture
def inputs():
    return ["This is a test complex sentence.", "This is also a test complex sentence."]

@pytest.fixture
def outputs():
    return ["This is a test sentence.", "This is also a test sentence."]

@pytest.fixture
def split_outputs():
    return ["This is a test. It is a sentence.", "This is also a test. It is a sentence."]


def test_split_acc(inputs, outputs, split_outputs):
    accs = calculate_split_acc(inputs, split_outputs, split_outputs)
    assert accs == True

    accs = calculate_split_acc(inputs, outputs, split_outputs)
    assert accs == False