from nltk import sent_tokenize

from control_simp.models.end_to_end import run_generator, BartFinetuner
from control_simp.models.classification import run_classifier, LightningBert


class RecursiveGenerator():

    def __init__(self, clf_loc, gen_loc, device="cuda"):
        self.device = device

        print("Loading classifier...")
        self.clf = LightningBert.load_from_checkpoint(clf_loc, model_type="roberta").to(device).eval()

        print("Loading generator...")
        self.gen = BartFinetuner.load_from_checkpoint(gen_loc, strict=False).to(device).eval()

    def generate(self, df, x_col="complex", k=1, max_samples=None):
        if max_samples is not None:
            df = df[:max_samples]

        for i in range(k):
            print(f"Iteration {i+1}...")
            step_pred = f"pred_{i+1}"

            # skip step if _i_th order predictions already exist
            if step_pred not in df.columns:
                print("Generating...")
                preds = []
                for i, row in df.iterrows():
                    # concatenate predicted simplification of each sentence in input
                    xs = sent_tokenize(row[x_col])
                    l_preds = run_classifier(self.clf, xs, device=self.device, return_logits=False)
                    ys = run_generator(self.gen, xs, ctrl_toks=l_preds)
                    preds.append(" ".join(ys))
                df[step_pred] = preds
            else:
                print("Generation already performed.")

            # treat this step's result as next step's input
            x_col = step_pred

        return df
