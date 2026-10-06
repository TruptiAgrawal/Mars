from slidesam_viz import *
d = load_tensorboard("../runs/sam/ddp/word_label2_fresh_4/tb")
for k, (s, v) in d.items():
    print(f"{k:20s} {len(s):6d} points   step {s.min()}–{s.max()}   last={v[-1]:.4f}")
