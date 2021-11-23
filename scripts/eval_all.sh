# wiki-auto end-to-end
python control_simp/models/eval.py bart /media/liam/data2/control_simp_ckps/365hhh2u/checkpoints/epoch\=3-step\=24175.ckpt \
/media/liam/data2/discourse_data/simp_clf_data/gen/control_simp_valid_exp1_clf.csv /media/liam/data2/control_simp_ckps/365hhh2u \
plain --samsa=False --ow=False

python control_simp/models/eval.py bart /media/liam/data2/control_simp_ckps/365hhh2u/checkpoints/epoch\=3-step\=24175.ckpt \
/media/liam/data2/simp_data/newsela-auto/newsela-auto/ACL2020/test_clf.csv /media/liam/data2/control_simp_ckps/365hhh2u \
newsela --samsa=False --ow=False

# newsela-auto end-to-end
python control_simp/models/eval.py bart /media/liam/data2/control_simp_ckps/3md1wk3q/checkpoints/epoch\=12-step\=53183.ckpt \
/media/liam/data2/discourse_data/simp_clf_data/gen/control_simp_valid_exp1_clf.csv /media/liam/data2/control_simp_ckps/3md1wk3q \
plain --samsa=False --ow=False

python control_simp/models/eval.py bart /media/liam/data2/control_simp_ckps/3md1wk3q/checkpoints/epoch\=12-step\=53183.ckpt \
/media/liam/data2/simp_data/newsela-auto/newsela-auto/ACL2020/test_clf.csv /media/liam/data2/control_simp_ckps/3md1wk3q \
newsela --samsa=False --ow=False

# 4-class end-to-end
python control_simp/models/eval.py bart /media/liam/data2/control_simp_ckps/3fn5qcza/checkpoints/epoch\=0.ckpt \
/media/liam/data2/discourse_data/simp_clf_data/gen/control_simp_valid_exp1_clf.csv /media/liam/data2/control_simp_ckps/3fn5qcza \
plain --samsa=False --ow=True

python control_simp/models/eval.py bart /media/liam/data2/control_simp_ckps/3fn5qcza/checkpoints/epoch\=0.ckpt \
/media/liam/data2/simp_data/newsela-auto/newsela-auto/ACL2020/test_clf.csv /media/liam/data2/control_simp_ckps/3fn5qcza \
newsela --samsa=False --ow=True

# 3-class end-to-end
python control_simp/models/eval.py bart /media/liam/data2/control_simp_ckps/d68v4v5z/checkpoints/epoch=3-step=259528.ckpt \
/media/liam/data2/discourse_data/simp_clf_data/gen/control_simp_valid_exp1_clf.csv /media/liam/data2/control_simp_ckps/d68v4v5z \
plain --samsa=False --ow=True

python control_simp/models/eval.py bart /media/liam/data2/control_simp_ckps/d68v4v5z/checkpoints/epoch=3-step=259528.ckpt \
/media/liam/data2/simp_data/newsela-auto/newsela-auto/ACL2020/test_clf.csv /media/liam/data2/control_simp_ckps/d68v4v5z \
newsela --samsa=False --ow=True

# 4-class ctrl-token
python control_simp/models/eval.py bart /media/liam/data2/control_simp_ckps/12s3nazz/checkpoints/epoch\=6-step\=166580.ckpt \
/media/liam/data2/discourse_data/simp_clf_data/gen/control_simp_valid_exp1_clf.csv /media/liam/data2/control_simp_ckps/12s3nazz/ \
clf --ctrl_toks=pred_l --samsa=False --ow=True

python control_simp/models/eval.py bart /media/liam/data2/control_simp_ckps/12s3nazz/checkpoints/epoch\=6-step\=166580.ckpt \
/media/liam/data2/discourse_data/simp_clf_data/gen/control_simp_valid_exp1_clf.csv /media/liam/data2/control_simp_ckps/12s3nazz/ \
oracle --ctrl_toks=label --samsa=False --ow=True

python control_simp/models/eval.py bart /media/liam/data2/control_simp_ckps/12s3nazz/checkpoints/epoch\=6-step\=166580.ckpt \
/media/liam/data2/simp_data/newsela-auto/newsela-auto/ACL2020/test_clf.csv /media/liam/data2/control_simp_ckps/12s3nazz \
newsela --ctrl_toks=pred_l --samsa=False --ow=True

python control_simp/models/eval.py bart /media/liam/data2/control_simp_ckps/12s3nazz/checkpoints/epoch\=6-step\=166580.ckpt \
/media/liam/data2/simp_data/newsela-auto/newsela-auto/ACL2020/test_clf.csv /media/liam/data2/control_simp_ckps/12s3nazz/ \
newsela_oracle --ctrl_toks=label --samsa=False --ow=True

# 3-class ctrl-token
python control_simp/models/eval.py bart /media/liam/data2/control_simp_ckps/25ldpr2h/checkpoints/epoch\=8-step\=170317.ckpt \
/media/liam/data2/discourse_data/simp_clf_data/gen/control_simp_valid_exp1_clf.csv /media/liam/data2/control_simp_ckps/25ldpr2h \
clf --ctrl_toks=pred_l --samsa=False --ow=True

python control_simp/models/eval.py bart /media/liam/data2/control_simp_ckps/25ldpr2h/checkpoints/epoch\=8-step\=170317.ckpt \
/media/liam/data2/discourse_data/simp_clf_data/gen/control_simp_valid_exp1_clf.csv /media/liam/data2/control_simp_ckps/25ldpr2h \
oracle --ctrl_toks=label --samsa=False --ow=True

python control_simp/models/eval.py bart /media/liam/data2/control_simp_ckps/25ldpr2h/checkpoints/epoch\=8-step\=170317.ckpt \
/media/liam/data2/simp_data/newsela-auto/newsela-auto/ACL2020/test_clf.csv /media/liam/data2/control_simp_ckps/25ldpr2h \
newsela --ctrl_toks=pred_l --samsa=False --ow=True

python control_simp/models/eval.py bart /media/liam/data2/control_simp_ckps/25ldpr2h/checkpoints/epoch\=8-step\=170317.ckpt \
/media/liam/data2/simp_data/newsela-auto/newsela-auto/ACL2020/test_clf.csv /media/liam/data2/control_simp_ckps/25ldpr2h \
newsela_oracle --ctrl_toks=label --samsa=False --ow=True

# s2s mtl 33% clf
python control_simp/models/eval.py bart /media/liam/data2/control_simp_ckps/1uehi0ry/checkpoints/epoch\=5-step\=150458.ckpt \
/media/liam/data2/discourse_data/simp_clf_data/gen/control_simp_valid_exp1_clf.csv /media/liam/data2/control_simp_ckps/1uehi0ry \
plain --samsa=False --ow=True --mtl=True

python control_simp/models/eval.py bart /media/liam/data2/control_simp_ckps/1uehi0ry/checkpoints/epoch\=5-step\=150458.ckpt\
/media/liam/data2/simp_data/newsela-auto/newsela-auto/ACL2020/test_clf.csv /media/liam/data2/control_simp_ckps/1uehi0ry \
newsela --samsa=False --ow=True --mtl=True

# s2s mtl 50% clf
python control_simp/models/eval.py bart /media/liam/data2/control_simp_ckps/22jcjfjs/checkpoints/epoch\=6-step\=188072.ckpt \
/media/liam/data2/discourse_data/simp_clf_data/gen/control_simp_valid_exp1_clf.csv /media/liam/data2/control_simp_ckps/22jcjfjs \
plain --samsa=False --ow=True --mtl=True

python control_simp/models/eval.py bart /media/liam/data2/control_simp_ckps/22jcjfjs/checkpoints/epoch\=6-step\=188072.ckpt \
/media/liam/data2/simp_data/newsela-auto/newsela-auto/ACL2020/test_clf.csv /media/liam/data2/control_simp_ckps/22jcjfjs \
newsela --samsa=False --ow=True --mtl=True