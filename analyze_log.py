#!/usr/bin/env python3
"""
GRPO 训练日志自动分析脚本
用法: python analyze_log.py [log_file]
默认分析 log_v7.log
"""

import re
import sys
import ast
from pathlib import Path
from collections import defaultdict


# ── 日志解析 ──────────────────────────────────────────────────────────────────

def parse_log(log_path: str) -> dict:
    """解析训练日志，返回结构化数据"""
    records = []
    header = {}
    errors = []
    current_step = 0
    max_steps = 0

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # 提取 header 信息
            m = re.search(r"Train samples:\s+(\d+)", line)
            if m:
                header["train_samples"] = int(m.group(1))
            m = re.search(r"Eval samples:\s+(\d+)", line)
            if m:
                header["eval_samples"] = int(m.group(1))
            m = re.search(r"model_name_or_path\s+(\S+)", line)
            if m:
                header["model"] = m.group(1)
            m = re.search(r"output_dir\s+(\S+)", line)
            if m:
                header["output_dir"] = m.group(1)

            # 提取当前进度
            m = re.search(r"(\d+)%\|.*?\s+(\d+)/(\d+)\s+\[", line)
            if m:
                current_step = int(m.group(2))
                max_steps = int(m.group(3))
            # 提取已训练时间 [4:23:35<13:42:20, ...]
            m = re.search(r"\[(\d+):(\d+):(\d+)<", line)
            if m:
                header["elapsed_hours"] = int(m.group(1)) + int(m.group(2)) / 60 + int(m.group(3)) / 3600

            # 提取错误
            if re.search(r"\b(Error|Exception|OOM|CUDA|Traceback)\b", line):
                errors.append(line.strip()[:200])

            # 提取 metric 行 — 可能在行首，也可能跟在进度条后面
            # 使用正则找到所有 {'loss': ...} 模式
            for m_obj in re.finditer(r"\{'loss':\s.*?\}", line):
                try:
                    raw_str = m_obj.group(0)
                    raw = ast.literal_eval(raw_str)
                    rec = {k: _to_float(v) for k, v in raw.items()}
                    records.append(rec)
                except Exception:
                    pass

    header["current_step"] = current_step
    header["max_steps"] = max_steps
    return {"header": header, "records": records, "errors": errors}


def _to_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


# ── 统计与趋势分析 ────────────────────────────────────────────────────────────

KEY_METRICS = [
    ("reward", "总奖励"),
    ("rewards/compute_score_em/mean", "EM (Exact Match)"),
    ("rewards/compute_score_f1/mean", "F1 Score"),
    ("rewards/compute_format_reward/mean", "格式奖励"),
    ("rewards/compute_search_persistence/mean", "搜索持续性"),
    ("kl", "KL 散度"),
    ("entropy", "策略熵"),
    ("loss", "Loss"),
    ("grad_norm", "梯度范数"),
    ("completions/mean_length", "平均输出长度"),
    ("frac_reward_zero_std", "零方差组比例"),
    ("step_time", "单步耗时(s)"),
    ("learning_rate", "学习率"),
]

TREND_KEYS = [
    "reward", "rewards/compute_score_em/mean", "rewards/compute_score_f1/mean",
    "kl", "entropy", "grad_norm", "completions/mean_length",
]


def analyze(records: list[dict]) -> dict:
    """对记录列表做统计分析"""
    if not records:
        return {}

    n = len(records)
    result = {}

    for key, label in KEY_METRICS:
        vals = [r[key] for r in records if key in r and isinstance(r[key], (int, float))]
        if not vals:
            continue

        # 移除 None / NaN
        clean = [v for v in vals if v == v]  # NaN != NaN
        if not clean:
            continue

        # 前 1/4 vs 后 1/4 趋势
        q1 = clean[: max(1, len(clean) // 4)]
        q4 = clean[-max(1, len(clean) // 4):]
        trend = "→"
        mean_q1 = _mean(q1)
        mean_q4 = _mean(q4)
        if mean_q4 > mean_q1 * 1.05:
            trend = "↑"
        elif mean_q4 < mean_q1 * 0.95:
            trend = "↓"

        # 检测异常值 (>2std)
        mean_all = _mean(clean)
        std_all = _std(clean)
        outliers = [v for v in clean if abs(v - mean_all) > 2 * std_all] if std_all > 0 else []

        # 移动平均 (窗口=5)
        ma = moving_average(clean, 5)

        result[key] = {
            "label": label,
            "first": clean[0],
            "last": clean[-1],
            "min": min(clean),
            "max": max(clean),
            "mean": mean_all,
            "std": std_all,
            "trend": trend,
            "outlier_count": len(outliers),
            "recent_5": clean[-5:] if len(clean) >= 5 else clean,
            "ma_last": ma[-1] if ma else clean[-1],  # 最新移动平均
        }

    return result


def moving_average(vals: list[float], window: int) -> list[float]:
    """计算移动平均"""
    if len(vals) < window:
        return [_mean(vals)]
    return [_mean(vals[i:i + window]) for i in range(len(vals) - window + 1)]


def detect_anomalies(records: list[dict], stats: dict) -> list[str]:
    """检测训练异常"""
    issues = []
    if not records:
        return ["无训练记录"]

    n = len(records)

    # 1. grad_norm 极低 (可能梯度消失)
    grad_vals = [r.get("grad_norm") for r in records if isinstance(r.get("grad_norm"), (int, float))]
    low_grad = [v for v in grad_vals if v < 0.1]
    if low_grad:
        issues.append(f"⚠ grad_norm 极低 (<0.1) 出现 {len(low_grad)} 次，可能梯度消失")

    # 2. KL 散度过大
    if "kl" in stats and stats["kl"]["last"] > 1.0:
        issues.append(f"⚠ KL 散度较大 (最新 {stats['kl']['last']:.3f})，策略偏离参考模型过多")
    elif "kl" in stats and stats["kl"]["last"] > 0.5:
        issues.append(f"⚡ KL 散度偏高 (最新 {stats['kl']['last']:.3f})，需持续关注")

    # 3. frac_reward_zero_std 偏高
    zero_std = [r.get("frac_reward_zero_std", 0) for r in records]
    recent_zero_std = zero_std[-10:] if len(zero_std) >= 10 else zero_std
    if _mean(recent_zero_std) > 0.4:
        issues.append(f"⚠ 近期零方差组比例偏高 ({_mean(recent_zero_std):.1%})，GRPO 优势估计质量下降")

    # 4. step_time 异常飙升
    times = [r.get("step_time") for r in records if isinstance(r.get("step_time"), (int, float))]
    if times:
        mean_t = _mean(times)
        spikes = [t for t in times if t > mean_t * 2]
        if spikes:
            issues.append(f"⚡ 单步耗时出现 {len(spikes)} 次异常飙升 (>{mean_t * 2:.0f}s)，可能与搜索API延迟有关")

    # 5. reward 停滞或下降
    if "reward" in stats:
        recent = stats["reward"]["recent_5"]
        if len(recent) >= 5:
            first_half = _mean(recent[:3])
            second_half = _mean(recent[2:])
            if second_half < first_half * 0.8:
                issues.append(f"⚠ 近期 reward 呈下降趋势 ({first_half:.3f} → {second_half:.3f})")

    # 6. entropy 过低
    if "entropy" in stats and stats["entropy"]["last"] < 0.15:
        issues.append(f"⚠ 策略熵过低 ({stats['entropy']['last']:.3f})，模型可能陷入模式坍塌")

    # 7. completion 长度骤变
    lens = [r.get("completions/mean_length") for r in records if isinstance(r.get("completions/mean_length"), (int, float))]
    if len(lens) >= 10:
        recent_lens = lens[-5:]
        prev_lens = lens[-10:-5]
        if abs(_mean(recent_lens) - _mean(prev_lens)) > _mean(prev_lens) * 0.4:
            issues.append(f"⚡ 输出长度近期变化较大 ({_mean(prev_lens):.0f} → {_mean(recent_lens):.0f})")

    if not issues:
        issues.append("✅ 未检测到明显异常")

    return issues


# ── 输出格式化 ─────────────────────────────────────────────────────────────────

def sparkline(values: list[float], width: int = 30) -> str:
    """生成 ASCII 迷你趋势图"""
    if not values or len(values) < 2:
        return ""
    chars = "▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    # 下采样到 width 个点
    step = max(1, len(values) // width)
    sampled = values[::step][:width]
    return "".join(chars[min(int((v - mn) / rng * (len(chars) - 1)), len(chars) - 1)] for v in sampled)


def fmt(v, decimals=4) -> str:
    if isinstance(v, float):
        if abs(v) < 0.001 and v != 0:
            return f"{v:.2e}"
        return f"{v:.{decimals}f}"
    return str(v)


def print_report(header: dict, stats: dict, anomalies: list[str], records: list[dict]):
    """打印分析报告"""
    n = len(records)

    # ── 基本信息 ──
    print("\n" + "=" * 72)
    print("  GRPO 训练日志分析报告")
    print("=" * 72)

    print(f"\n📋 基本信息")
    print(f"  训练/验证集:  {header.get('train_samples', '?')} / {header.get('eval_samples', '?')}")
    print(f"  当前进度:     {header.get('current_step', '?')} / {header.get('max_steps', '?')} 步 "
          f"({header.get('current_step', 0) / max(header.get('max_steps', 1), 1):.1%})")

    if n > 0:
        # step_time 是每个 logging_step 的总耗时（包含多个子步）
        # 实际总步数 = current_step, 但 metric 只记录了 n 次
        # 用进度条信息估算已训练时间
        avg_time = _mean([r.get("step_time", 0) for r in records if isinstance(r.get("step_time"), (int, float))])
        current = header.get("current_step", 0)
        # 粗估总时间 = 当前步数 * 平均每步耗时
        # 但 step_time 是 logging_step 的耗时，不是每步
        # 用更精确的方式: 从进度条提取已训练时间
        elapsed_h = header.get("elapsed_hours", 0)
        if elapsed_h == 0 and avg_time > 0:
            # fallback: 用 current_step * avg_time_per_actual_step
            # 从 step_time 估算: 每个 metric 记录对应若干步
            steps_per_metric = current / max(n, 1)
            elapsed_h = current * avg_time / steps_per_metric / 3600
        remaining_steps = header.get("max_steps", 0) - current
        eta_h = remaining_steps * (elapsed_h * 3600 / max(current, 1)) / 3600 if current > 0 else 0
        print(f"  已训练时间:   ~{elapsed_h:.1f}h")
        print(f"  平均步耗时:   {avg_time:.1f}s (每 logging_step)")
        print(f"  预计剩余:     ~{eta_h:.1f}h")
        print(f"  日志记录数:   {n} 条 (每 num_generations×batch 记录一次)")

    # ── 核心指标 ──
    print(f"\n📊 核心指标趋势")
    print(f"{'指标':<20} {'初始':>10} {'最新':>10} {'最小':>10} {'最大':>10} {'均值':>10} {'趋势':>4}  {'迷你图'}")
    print("-" * 100)

    # 按 KEY_METRICS 顺序输出
    for key, label in KEY_METRICS:
        if key not in stats:
            continue
        s = stats[key]
        vals = [r[key] for r in records if key in r and isinstance(r[key], (int, float))]
        chart = sparkline(vals, 25) if len(vals) >= 5 else ""
        decimals = 2 if key in ("step_time", "completions/mean_length") else 4
        print(f"  {label:<18} {fmt(s['first'], decimals):>10} {fmt(s['last'], decimals):>10} "
              f"{fmt(s['min'], decimals):>10} {fmt(s['max'], decimals):>10} {fmt(s['mean'], decimals):>10} "
              f"  {s['trend']:>2}   {chart}")

    # ── 分项奖励拆解 ──
    if records:
        print(f"\n📈 分项奖励拆解 (最新5条记录)")
        sub_keys = [
            ("rewards/compute_score_em/mean", "EM"),
            ("rewards/compute_score_f1/mean", "F1"),
            ("rewards/compute_format_reward/mean", "格式"),
            ("rewards/compute_search_persistence/mean", "搜索持续性"),
        ]
        print(f"  {'Step':>6}  {'总Reward':>9}  " + "  ".join(f"{lbl:>8}" for _, lbl in sub_keys))
        print("  " + "-" * 60)
        for r in records[-5:]:
            step_str = f"{r.get('epoch', 0):.4f}"
            parts = [f"{r.get('reward', 0):.4f}"]
            for k, _ in sub_keys:
                v = r.get(k, 0)
                parts.append(f"{v:.4f}" if isinstance(v, (int, float)) else "N/A")
            print(f"  {step_str:>6}  " + "  ".join(f"{p:>9}" for p in parts))

    # ── 异常检测 ──
    print(f"\n🔍 异常检测")
    for a in anomalies:
        print(f"  {a}")

    # ── 训练阶段总结 ──
    if n >= 5:
        print(f"\n📋 训练阶段总结")
        _print_phase_summary(records, stats)

    # ── 建议 ──
    print(f"\n💡 训练建议")
    suggestions = _generate_suggestions(stats, records)
    for s in suggestions:
        print(f"  • {s}")

    print("\n" + "=" * 72)


def _generate_suggestions(stats: dict, records: list[dict]) -> list[str]:
    suggestions = []

    if "reward" in stats:
        s = stats["reward"]
        if s["last"] < 0.1:
            suggestions.append("总奖励很低，考虑检查 reward 函数设计或增加训练步数")
        elif s["trend"] == "↑":
            suggestions.append("奖励持续上升趋势，建议继续训练")
        elif s["trend"] == "↓":
            suggestions.append("奖励呈下降趋势，可能需要调整学习率或检查 reward 设计")

    if "kl" in stats:
        s = stats["kl"]
        if s["last"] > 0.5:
            suggestions.append("KL散度偏高，可考虑增大 beta (当前0.02) 到 0.04~0.05 以限制策略偏移")
        if s["trend"] == "↑" and s["last"] > 0.3:
            suggestions.append("KL散度持续增长中，需关注是否出现奖励投机(reward hacking)")

    zero_std_vals = [r.get("frac_reward_zero_std", 0) for r in records[-10:]]
    if _mean(zero_std_vals) > 0.3:
        suggestions.append("零方差组比例偏高，部分组内所有回答得分相同，GRPO无法区分优劣；"
                          "可考虑增加 num_generations 或优化 reward 细粒度")

    grad_vals = [r.get("grad_norm") for r in records if isinstance(r.get("grad_norm"), (int, float))]
    low_grad_count = sum(1 for v in grad_vals[-10:] if v < 0.1)
    if low_grad_count > 2:
        suggestions.append("近期梯度范数频繁极低，检查是否存在 reward 全同导致梯度消失的情况")

    if "rewards/compute_format_reward/mean" in stats:
        s = stats["rewards/compute_format_reward/mean"]
        if s["last"] >= s["max"] * 0.99:
            suggestions.append("格式奖励已达到满分，模型格式学习完成")

    if not suggestions:
        suggestions.append("训练状态良好，继续当前配置即可")

    return suggestions


def _print_phase_summary(records: list[dict], stats: dict):
    """打印训练阶段总结（早期/中期/晚期）"""
    n = len(records)
    third = max(1, n // 3)
    phases = [
        ("(前1/3)", records[:third]),
        ("(中1/3)", records[third:2 * third]),
        ("(后1/3)", records[2 * third:]),
    ]

    keys = [
        ("reward", "Reward"),
        ("rewards/compute_score_em/mean", "EM"),
        ("rewards/compute_score_f1/mean", "F1"),
        ("kl", "KL"),
        ("entropy", "Entropy"),
        ("grad_norm", "GradNorm"),
    ]

    print(f"  {'指标':<12}", end="")
    for phase_name, _ in phases:
        print(f"  {phase_name:>14}", end="")
    print()
    print("  " + "-" * 56)

    for key, label in keys:
        print(f"  {label:<12}", end="")
        for _, phase_records in phases:
            vals = [r[key] for r in phase_records if key in r and isinstance(r[key], (int, float))]
            if vals:
                print(f"  {_mean(vals):>14.4f}", end="")
            else:
                print(f"  {'N/A':>14}", end="")
        print()


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _mean(vals):
    return sum(vals) / len(vals) if vals else 0


def _std(vals):
    if len(vals) < 2:
        return 0
    m = _mean(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    log_file = sys.argv[1] if len(sys.argv) > 1 else "log_v7.log"
    log_path = Path(log_file)

    if not log_path.exists():
        print(f"错误: 日志文件 {log_file} 不存在")
        sys.exit(1)

    size_mb = log_path.stat().st_size / 1024 / 1024
    print(f"正在解析日志: {log_file} ({size_mb:.1f}MB) ...")

    data = parse_log(str(log_path))
    stats = analyze(data["records"])
    anomalies = detect_anomalies(data["records"], stats)
    print_report(data["header"], stats, anomalies, data["records"])

    if data["errors"]:
        print(f"\n❌ 检测到 {len(data["errors"])} 条错误/警告:")
        for e in data["errors"][:10]:
            print(f"  {e[:150]}")


if __name__ == "__main__":
    main()
