
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
        df["pred"] = df[x_col]

        for i in range(k):
            print(f"Iteration {i}...")
            print("Predicting labels...")
            df["pred_l"] = run_classifier(self.clf, df, "pred", max_samples=max_samples, device=self.device, return_logits=False)

            print("Generating predictions...")
            df["pred"] = run_generator(self.gen, df, ctrl_toks="pred_l", max_samples=max_samples)

        return df
