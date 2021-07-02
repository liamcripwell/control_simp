import itertools
from typing import Callable, Iterable, List


def lmap(f: Callable, x: Iterable) -> List:
    return list(map(f, x))


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]