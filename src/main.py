"""Interactive launcher, same idea as FGC-RVFL/src/main.py: pick the experiment then run."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "models"))

from experiment import run, SEEDS, OUT  # noqa: E402


def ask_int(prompt, valid):
    while True:
        try:
            choice = int(input(prompt))
            if choice in valid:
                return choice
            print(f"请输入有效的选项数字 ({', '.join(str(v) for v in valid)})")
        except ValueError:
            print("请输入有效的数字")


def pick_seeds():
    print("\n请选择种子:")
    print(f"1. 全部 {SEEDS}（从 1..10000 抽 5 个，已写死）")
    print(f"2. 试跑一个种子 ({SEEDS[0]})")
    choice = ask_int("请输入选项数字 (1 或 2): ", {1, 2})
    return SEEDS if choice == 1 else [SEEDS[0]]


def main():
    print("请选择实验类型:")
    print("1. 论文主实验 (25 个多分类表格任务)")
    print("2. 指定任务")
    print("3. 汇总已有结果 (不跑实验)")

    kind = ask_int("请输入选项数字 (1, 2 或 3): ", {1, 2, 3})

    if kind == 3:
        from analyze import main as analyze_main
        analyze_main()
        return

    seeds = pick_seeds()

    if kind == 1:
        names, out = None, OUT
        label = "论文主实验（25 个多分类）"
    else:
        raw = input("请输入任务名（逗号分隔，例如 Wine）: ").strip()
        names = set(s.strip() for s in raw.split(",") if s.strip())
        if not names:
            raise SystemExit("没有输入任务名")
        out = input(f"结果 CSV 路径（回车则 {OUT}）: ").strip() or OUT
        label = "指定任务 " + ", ".join(sorted(names))

    print(f"\n实验: {label}")
    print(f"种子: {seeds}")
    print(f"输出: {out}")
    print("开始前会清空 results/ 下的旧结果。")
    run(out, seeds, names)


if __name__ == "__main__":
    main()
