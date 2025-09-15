# plot_cheetah.py
import os
import glob
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# --------------------------
# 路径解析：优先用 cs285/data，其次用 data
# --------------------------
def resolve_data_dir() -> str:
    candidates = ["cs285/data", "data"]
    for d in candidates:
        if os.path.isdir(d):
            return d
    # 若都不存在，就用默认 "data" 并创建
    os.makedirs("data", exist_ok=True)
    return "data"

DATA_DIR = resolve_data_dir()

# --------------------------
# 读 tfevents（取最近修改的那个）
# --------------------------
def newest_tfevent(run_dir: str):
    files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]

def load_scalar(tbe_path: str, tag: str):
    """返回 (steps, values)；若不存在该 tag，返回空列表"""
    ea = event_accumulator.EventAccumulator(tbe_path)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    evs = ea.Scalars(tag)
    return [e.step for e in evs], [e.value for e in evs]

def load_vals_only(tbe_path: str, tag: str):
    _, y = load_scalar(tbe_path, tag)
    return y

# --------------------------
# 精确匹配最新 run：q2_pg_{exp_name}_{env_name}_timestamp
# --------------------------
def find_latest_run(exp_name: str, env_name: str):
    prefix = f"q2_pg_{exp_name}_{env_name}_"
    candidates = [
        d for d in os.listdir(DATA_DIR)
        if d.startswith(prefix) and os.path.isdir(os.path.join(DATA_DIR, d))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda d: os.path.getmtime(os.path.join(DATA_DIR, d)))
    return candidates[-1]

# --------------------------
# 画 Baseline Loss
# --------------------------
def plot_baseline_loss(env_name: str = "HalfCheetah-v4", exp_name: str = "cheetah_baseline"):
    run = find_latest_run(exp_name=exp_name, env_name=env_name)
    if not run:
        print(f"[WARN] 找不到 {exp_name} 的日志目录（环境 {env_name}）")
        return
    run_dir = os.path.join(DATA_DIR, run)
    tbe = newest_tfevent(run_dir)
    if not tbe:
        print(f"[WARN] {run} 没有 tfevents 文件")
        return

    # y = Baseline Loss；x 用 Train_EnvstepsSoFar（若缺失则退回迭代步）
    x_iter, y = load_scalar(tbe, "Baseline Loss")
    if not x_iter:
        print(f"[WARN] {run} 没有记录 Baseline Loss")
        return

    x_env = load_vals_only(tbe, "Train_EnvstepsSoFar")
    x = x_env[:len(y)] if x_env else x_iter

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Env Steps (Train_EnvstepsSoFar)" if x_env else "Iteration")
    plt.ylabel("Baseline Loss")
    plt.title(f"{env_name} ({exp_name}) - Baseline Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = "cheetah_baseline_loss.png"
    plt.savefig(out, dpi=200)
    print(f"[OK] 保存 {out}")

# --------------------------
# 画 Eval_AverageReturn 对比
# --------------------------
def plot_eval_return_compare(env_name: str = "HalfCheetah-v4", with_na: bool = False):
    runs = {
        "no_baseline": ("cheetah", env_name),
        "baseline": ("cheetah_baseline", env_name),
    }
    if with_na:
        runs["baseline+na"] = ("cheetah_baseline_na", env_name)

    plt.figure()
    any_line = False

    for label, (exp_name, env) in runs.items():
        run = find_latest_run(exp_name, env)
        if not run:
            print(f"[WARN] 找不到 {label} 对应的 run：q2_pg_{exp_name}_{env}_*")
            continue

        run_dir = os.path.join(DATA_DIR, run)
        tbe = newest_tfevent(run_dir)
        if not tbe:
            print(f"[WARN] {run} 没有 tfevents 文件")
            continue

        x_iter, y = load_scalar(tbe, "Eval_AverageReturn")
        if not x_iter:
            print(f"[WARN] {run} 缺少 Eval_AverageReturn")
            continue

        # 横轴：Train_EnvstepsSoFar（优先），兜底：迭代步
        x_env = load_vals_only(tbe, "Train_EnvstepsSoFar")
        x = x_env[:len(y)] if x_env else x_iter

        plt.plot(x, y, label=label)
        any_line = True

    if not any_line:
        print("[WARN] 没有画出任何曲线（可能日志缺失或 tag 不存在）")
        return

    plt.xlabel("Env Steps (Train_EnvstepsSoFar)" if any_line else "Iteration")
    plt.ylabel("Eval_AverageReturn")
    title = f"{env_name}: with vs without baseline" + (" (+na)" if with_na else "")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = "cheetah_eval_compare.png" if not with_na else "cheetah_eval_compare_with_na.png"
    plt.savefig(out, dpi=200)
    print(f"[OK] 保存 {out}")

# --------------------------
# CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HalfCheetah-v4",
                        help="环境名，默认 HalfCheetah-v4")
    parser.add_argument("--with_na", action="store_true",
                        help="同时对比 baseline+na（需要先跑过 --exp_name cheetah_baseline_na）")
    parser.add_argument("--skip_loss", action="store_true",
                        help="只画 Eval_AverageReturn 对比，不画 Baseline Loss")
    args = parser.parse_args()

    if not args.skip_loss:
        plot_baseline_loss(env_name=args.env, exp_name="cheetah_baseline")
    plot_eval_return_compare(env_name=args.env, with_na=args.with_na)

if __name__ == "__main__":
    main()