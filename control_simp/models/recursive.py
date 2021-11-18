from nltk import sent_tokenize

from control_simp.utils import flatten_list
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

                # compile set of individual sentences and C idx
                xs = []
                cids = []
                for i, x in df[x_col].iteritems():
                    sents = sent_tokenize(x)
                    xs.append(sents)
                    cids += [i]*len(sents)
                xs = flatten_list(xs)

                # run prediction and generation models
                ls = run_classifier(self.clf, xs, device=self.device, return_logits=False)
                ys = run_generator(self.gen, xs, ctrl_toks=ls)

                # rebuild full outputs from individual sentences
                pred_ls = []
                preds = []
                l_buff = []
                y_buff = []
                for xid, cid in enumerate(cids):
                    if cid != cids[xid-1]:
                        pred_ls.append(l_buff)
                        preds.append(" ".join(y_buff))
                    else:
                        l_buff = []
                        y_buff = []
                    l_buff.append(ls[xid])
                    # forceably use inputs where 0 label predicted
                    if ls[xid] == 0:
                        y_buff.append(xs[xid].replace("<ident> ", ""))
                    else:
                        y_buff.append(ys[xid])
                pred_ls.append(l_buff)
                preds.append(" ".join(y_buff))
                assert len(pred_ls) == len(df)
                assert len(preds) == len(df)

                # add predictions to dataframe
                df[f"labels_{i+1}"] = pred_ls
                df[step_pred] = preds
            else:
                print("Generation already performed.")

            # treat this step's result as next step's input
            x_col = step_pred

        return df

    def __getattr__(self, name):
        if name == "tokenizer":
            return self.gen.tokenizer
        else:
            raise AttributeError
