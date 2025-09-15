import os
import re
import glob
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def load_scalar(path, tag):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    vals  = [e.value for e in events]
    return steps, vals

def find_event_file(run_dir):
    # 取该目录下第一个 tfevents 文件
    files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    return files[0] if files else None

def plot_group(data_root, prefix, title, outfile):
    """
    prefix:
      - 小 batch 组:   'q2_pg_cartpole_'         (不含 lb)
      - 大 batch 组:   'q2_pg_cartpole_lb_'      (带 lb)
    """
    # 自动收集该组所有实验
    runs = []
    for d in sorted(os.listdir(data_root)):
        if d.startswith(prefix):
            runs.append(d)
    if not runs:
        print(f"[WARN] No runs with prefix {prefix} in {data_root}")
        return

    plt.figure()
    for run in runs:
        run_dir = os.path.join(data_root, run)
        ev = find_event_file(run_dir)
        if ev is None:
            print(f"[WARN] No tfevents in {run_dir}")
            continue

        # 横轴要求：Train_EnvstepsSoFar
        steps, rets = load_scalar(ev, "Train_AverageReturn")
        x,     _    = load_scalar(ev, "Train_EnvstepsSoFar")
        if not steps or not x:
            print(f"[WARN] Missing scalars in {run}")
            continue

        # 生成更干净的label
        # 从 exp_name 中截取 cartpole 子串后面的标签，比如 cartpole_rtg_na
        m = re.search(r'(cartpole.*)', run)
        label = m.group(1) if m else run

        plt.plot(x, rets, label=label)

    plt.xlabel("Env Steps (Train_EnvstepsSoFar)")
    plt.ylabel("Train_AverageReturn")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    print(f"[OK] Saved {outfile}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="path to hw2/data")
    args = ap.parse_args()

    # 小 & 大 batch 两张图
    plot_group(args.data,
               prefix="q2_pg_cartpole_",
               title="CartPole PG (small batch)",
               outfile="cartpole_small_batch.png")

    plot_group(args.data,
               prefix="q2_pg_cartpole_lb_",
               title="CartPole PG + baseline (large batch)",
               outfile="cartpole_large_batch.png")

if __name__ == "__main__":
    main()
    
    

#– Which value estimator has better performance without advantage normalization: the trajectory - centric one, or the one using reward-to-go?
# The trajectory-centric one.

#– Did advantage normalization help?
# Yes

#– Did the batch size make an impact?
# Yes, larger batch size helps