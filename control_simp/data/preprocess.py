import json

import fire
import pandas as pd

from control_simp.data.bart import pretokenize as bart_pretok
from control_simp.data.bert import pretokenize as bert_pretok


class Launcher(object):

    def bart(model, data_file, output_dir, x_col, y_col, max_samples):
        df = pd.read_csv(data_file)
        bart_pretok(model, df, output_dir, x_col, y_col, max_samples)
        meta = {
            "data_file": data_file,
            "x_col": x_col,
            "y_col": y_col,
            "max_samples": max_samples
        }
        json.dump(meta, open(f"{output_dir}/meta.json", "w"))


if __name__ == '__main__':
    fire.Fire(Launcher)