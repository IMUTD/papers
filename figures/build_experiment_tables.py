"""Export experiment CSVs from logged summaries, then draw figures from those CSVs.

Numbers come from existing runs. Nothing here retrains a model.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from sklearn.decomposition import PCA

ROOT = Path(r"e:\WPSCloud\iepe\workspace")
OUT = ROOT / "papers" / "tables"
FIG = ROOT / "papers" / "figures"
OURS = ROOT / "OpenFCIFD" / "outputs"
BASE = ROOT / "sysbaseline" / "task-sys" / "outputs"

OURS_RUN = {
    "TE": "te_imbalanced_openworld_sclifd_gmm_raw_mix_kmraw_keepnew_joint_sil",
    "MFF": "mff_imbalanced_openworld_sclifd_gmm_raw_mix_kmraw_keepnew_joint_sil",
}
ABLATION = {
    "硬门整包": "gmm_raw_mix",
    "固定K锚定": "gmm_raw_mix_kmraw_keepnew_joint",
    "固定K停止更新": "gmm_raw_mix_kmraw_keepnew_joint_freeze",
    "本文": "gmm_raw_mix_kmraw_keepnew_joint_sil",
}
BASELINES = [
    ("PromptCCD", "promptccd"),
    ("IGCD-u", "igcd"),
    ("NCD-DLT", "ncddlt"),
    ("OpenGCD", "opengcd"),
]
POLLUTED = {"TE": "1", "MFF": "0"}

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ours_summary(dataset: str, suffix: str) -> dict:
    prefix = "te" if dataset == "TE" else "mff"
    path = OURS / f"{prefix}_imbalanced_openworld_sclifd_{suffix}" / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def base_summary(slug: str, dataset: str) -> dict:
    path = BASE / f"{slug}_estk_{dataset.lower()}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def acc_of(per_class: dict, key: str):
    item = per_class.get(key)
    if item is None:
        return ""
    if isinstance(item, dict):
        return item.get("acc", "")
    return item


def n_of(per_class: dict, key: str):
    item = per_class.get(key)
    if isinstance(item, dict):
        return item.get("n", "")
    return ""


def export_split() -> None:
    rows = []
    files = {
        "TE": ROOT / "sysbaseline" / "te_sessions.npz",
        "MFF": ROOT / "sysbaseline" / "mff_sessions.npz",
    }
    for dataset, path in files.items():
        z = np.load(path, allow_pickle=True)
        n_sess = 5 if dataset == "TE" else 3
        for t in range(n_sess):
            y = z[f"s{t}_y"]
            te = z[f"s{t}_test_y"]
            labels, counts = np.unique(y, return_counts=True)
            train_counts = {int(a): int(b) for a, b in zip(labels, counts)}
            if t == 0:
                n_seen = int(len(y))
                n_new = 0
            else:
                seen = set(int(c) for c in z[f"s{t}_seen"].tolist())
                n_seen = int(sum(train_counts[c] for c in train_counts if c in seen))
                n_new = int(len(y) - n_seen)
            rows.append(
                {
                    "dataset": dataset,
                    "session": t,
                    "n_train": int(len(y)),
                    "n_seen_in_train": n_seen,
                    "n_new_in_train": n_new,
                    "seen_ratio": round(n_seen / len(y), 4),
                    "train_class_counts": json.dumps(train_counts, ensure_ascii=False),
                    "n_test": int(len(te)),
                    "test_classes": int(len(np.unique(te))),
                    "labels_used_in_training": "yes" if t == 0 else "no",
                }
            )
    write_csv(OUT / "split.csv", rows)


def export_session_and_per_class() -> None:
    session_rows = []
    final_rows = []
    heat_rows = []
    reg_rows = []
    for dataset in ("TE", "MFF"):
        data = ours_summary(dataset, OURS_RUN[dataset].split("sclifd_")[1])
        sessions = data["sessions"]
        for s in sessions:
            t = int(s["session"])
            session_rows.append(
                {
                    "method": "本文",
                    "dataset": dataset,
                    "session": t,
                    "test_acc": s["test_acc"],
                    "input": "minmax_vector",
                }
            )
            per = {str(k): v for k, v in s["per_class"].items()}
            for c, item in sorted(per.items(), key=lambda kv: int(kv[0])):
                heat_rows.append(
                    {
                        "method": "本文",
                        "dataset": dataset,
                        "session": t,
                        "class": int(c),
                        "acc": item["acc"],
                        "n": item["n"],
                    }
                )
            if t == len(sessions) - 1:
                for c, item in sorted(per.items(), key=lambda kv: int(kv[0])):
                    final_rows.append(
                        {
                            "method": "本文",
                            "dataset": dataset,
                            "class": int(c),
                            "acc": item["acc"],
                            "n": item["n"],
                            "input": "minmax_vector",
                        }
                    )
            leak = s.get("arrive_id_leak")
            reg_rows.append(
                {
                    "dataset": dataset,
                    "session": t,
                    "n_train_arrive": s.get("n_train_arrive", ""),
                    "n_arrive_id": s.get("n_arrive_id", ""),
                    "n_arrive_new": s.get("n_arrive_new", ""),
                    "arrive_id_leak": "" if leak is None else leak,
                    "n_registered_new": s.get("n_registered_new", ""),
                    "n_true_new": s.get("n_true_new", ""),
                    "n_false_new": s.get("n_false_new", ""),
                    "polluted_class": POLLUTED[dataset],
                    "polluted_acc": acc_of(per, POLLUTED[dataset]),
                    "class3_acc": acc_of(per, "3"),
                    "class3_n": n_of(per, "3"),
                    "test_acc": s["test_acc"],
                }
            )
        for label, slug in BASELINES:
            b = base_summary(slug, dataset)
            for s in b["sessions"]:
                t = int(s["session"]) if "session" in s else b["sessions"].index(s)
                session_rows.append(
                    {
                        "method": label,
                        "dataset": dataset,
                        "session": t,
                        "test_acc": s["test_acc"],
                        "input": "raster224",
                    }
                )
            last = b["sessions"][-1]
            per = {str(k): v for k, v in last["per_class"].items()}
            for c, item in sorted(per.items(), key=lambda kv: int(kv[0])):
                acc = item["acc"] if isinstance(item, dict) else item
                n = item.get("n", "") if isinstance(item, dict) else ""
                final_rows.append(
                    {
                        "method": label,
                        "dataset": dataset,
                        "class": int(c),
                        "acc": acc,
                        "n": n,
                        "input": "raster224",
                    }
                )
    write_csv(OUT / "session_acc.csv", session_rows)
    write_csv(OUT / "final_per_class.csv", final_rows)
    write_csv(OUT / "ours_per_class_by_session.csv", heat_rows)
    write_csv(OUT / "registration.csv", reg_rows)


def export_ablation() -> None:
    rows = []
    for dataset in ("TE", "MFF"):
        prefix = "te" if dataset == "TE" else "mff"
        last_t = 4 if dataset == "TE" else 2
        for name, suffix in ABLATION.items():
            data = ours_summary(dataset, suffix)
            sessions = data["sessions"]
            by_t = {int(s["session"]): s for s in sessions}
            s1 = by_t[1]
            last = by_t[last_t]
            per1 = {str(k): v for k, v in s1["per_class"].items()}
            perL = {str(k): v for k, v in last["per_class"].items()}
            rows.append(
                {
                    "setting": name,
                    "dataset": dataset,
                    "s1_leak": s1.get("arrive_id_leak", ""),
                    "polluted_class": POLLUTED[dataset],
                    "polluted_acc_s1": acc_of(per1, POLLUTED[dataset]),
                    "final_session": last_t,
                    "final_test_acc": last["test_acc"],
                    "class3_acc_final": acc_of(perL, "3"),
                    "false_new_final": last.get("n_false_new", ""),
                    "mean_test_acc": sum(s["test_acc"] for s in sessions) / len(sessions),
                }
            )
    write_csv(OUT / "ablation.csv", rows)


def export_overlap() -> None:
    """PCA of the final test set. Labels color points; they are not model outputs."""
    specs = {
        "TE": (ROOT / "sysbaseline" / "te_sessions.npz", 4, (0, 8)),
        "MFF": (ROOT / "sysbaseline" / "mff_sessions.npz", 2, (0, 4)),
    }
    rows = []
    for dataset, (path, t, pair) in specs.items():
        z = np.load(path, allow_pickle=True)
        x = z[f"s{t}_test_x"]
        y = z[f"s{t}_test_y"].astype(int)
        pca = PCA(n_components=2, random_state=66)
        xy = pca.fit_transform(x)
        var = pca.explained_variance_ratio_
        for i in range(len(y)):
            if int(y[i]) not in pair and int(y[i]) != pair[0]:
                # keep the polluted old class and the later class named in the pair,
                # plus every class so the table can be refiltered
                pass
            rows.append(
                {
                    "dataset": dataset,
                    "session": t,
                    "class": int(y[i]),
                    "pc1": f"{xy[i, 0]:.6f}",
                    "pc2": f"{xy[i, 1]:.6f}",
                    "pc1_var": f"{var[0]:.6f}",
                    "pc2_var": f"{var[1]:.6f}",
                    "focus_old": pair[0],
                    "focus_late": pair[1],
                }
            )
    write_csv(OUT / "final_test_pca.csv", rows)


def plot_lines() -> None:
    rows = read_csv(OUT / "session_acc.csv")
    methods = ["本文", "PromptCCD", "IGCD-u", "NCD-DLT", "OpenGCD"]
    colors = {
        "本文": "#111111",
        "PromptCCD": "#4C78A8",
        "IGCD-u": "#F58518",
        "NCD-DLT": "#54A24B",
        "OpenGCD": "#E45756",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, dataset, title in zip(axes, ("TE", "MFF"), ("TE，五轮", "MFF，三轮")):
        for method in methods:
            pts = [
                r
                for r in rows
                if r["dataset"] == dataset and r["method"] == method
            ]
            pts = sorted(pts, key=lambda r: int(r["session"]))
            xs = [int(r["session"]) for r in pts]
            ys = [float(r["test_acc"]) * 100 for r in pts]
            ax.plot(
                xs,
                ys,
                marker="o",
                color=colors[method],
                linewidth=2.0 if method == "本文" else 1.3,
                label=method,
            )
        ax.set_title(title)
        ax.set_xlabel("会话")
        ax.set_ylabel("测试准确率 / %")
        ax.set_xticks(xs)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "05_session_acc.png")
    plt.close(fig)


def plot_bars() -> None:
    rows = [r for r in read_csv(OUT / "final_per_class.csv") if r["method"] == "本文"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, dataset in zip(axes, ("TE", "MFF")):
        pts = sorted(
            [r for r in rows if r["dataset"] == dataset],
            key=lambda r: int(r["class"]),
        )
        xs = [int(r["class"]) for r in pts]
        ys = [float(r["acc"]) * 100 for r in pts]
        ax.bar(xs, ys, color="#4C78A8")
        ax.set_ylim(0, 105)
        ax.set_xlabel("类别编号")
        ax.set_ylabel("最终准确率 / %")
        ax.set_title(f"{dataset} 最终逐类")
        ax.set_xticks(xs)
    fig.tight_layout()
    fig.savefig(FIG / "06_final_per_class.png")
    plt.close(fig)


def plot_ablation() -> None:
    rows = read_csv(OUT / "ablation.csv")
    order = ["硬门整包", "固定K锚定", "固定K停止更新", "本文"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    metrics = [
        ("s1_leak", "s1 已见漏入 / %"),
        ("final_test_acc", "最终测试准确率 / %"),
    ]
    for ax, (key, title) in zip(axes, metrics):
        width = 0.35
        x = np.arange(len(order))
        for offset, dataset, color in ((-width / 2, "TE", "#4C78A8"), (width / 2, "MFF", "#F58518")):
            vals = []
            for name in order:
                hit = next(r for r in rows if r["setting"] == name and r["dataset"] == dataset)
                vals.append(float(hit[key]) * 100)
            ax.bar(x + offset, vals, width=width, label=dataset, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=15, ha="right")
        ax.set_ylabel(title)
        ax.set_title(title)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "07_ablation.png")
    plt.close(fig)


def plot_heat() -> None:
    rows = read_csv(OUT / "ours_per_class_by_session.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), constrained_layout=True)
    im = None
    for ax, dataset, n_s, n_c in zip(axes, ("TE", "MFF"), (5, 3), (10, 6)):
        grid = np.full((n_c, n_s), np.nan)
        for r in rows:
            if r["dataset"] != dataset:
                continue
            grid[int(r["class"]), int(r["session"])] = float(r["acc"]) * 100
        im = ax.imshow(grid, vmin=0, vmax=100, cmap="Blues", aspect="auto")
        ax.set_xticks(range(n_s))
        ax.set_xticklabels([f"s{t}" for t in range(n_s)])
        ax.set_yticks(range(n_c))
        ax.set_title(dataset)
        ax.set_xlabel("会话")
        ax.set_ylabel("类别")
        for i in range(n_c):
            for j in range(n_s):
                val = grid[i, j]
                if np.isnan(val):
                    continue
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=axes, shrink=0.85, label="准确率 / %")
    fig.savefig(FIG / "08_per_class_heatmap.png")
    plt.close(fig)


def plot_overlap() -> None:
    rows = read_csv(OUT / "final_test_pca.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for ax, dataset in zip(axes, ("TE", "MFF")):
        pts = [r for r in rows if r["dataset"] == dataset]
        old = int(pts[0]["focus_old"])
        late = int(pts[0]["focus_late"])
        var1 = float(pts[0]["pc1_var"]) * 100
        var2 = float(pts[0]["pc2_var"]) * 100
        for cls, color, marker, name in (
            (old, "#4C78A8", "o", f"类 {old}"),
            (late, "#E45756", "^", f"类 {late}"),
        ):
            sub = [r for r in pts if int(r["class"]) == cls]
            ax.scatter(
                [float(r["pc1"]) for r in sub],
                [float(r["pc2"]) for r in sub],
                s=8,
                alpha=0.35,
                c=color,
                marker=marker,
                label=name,
                linewidths=0,
            )
        ax.set_title(f"{dataset} 最终测试")
        ax.set_xlabel(f"PC1 ({var1:.1f}%)")
        ax.set_ylabel(f"PC2 ({var2:.1f}%)")
        ax.legend(frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(FIG / "09_late_overlap.png")
    plt.close(fig)


def plot_registration() -> None:
    rows = read_csv(OUT / "registration.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, dataset in zip(axes, ("TE", "MFF")):
        pts = [r for r in rows if r["dataset"] == dataset and int(r["session"]) > 0]
        xs = [int(r["session"]) for r in pts]
        leak = [float(r["arrive_id_leak"]) * 100 for r in pts]
        c3 = [float(r["class3_acc"]) * 100 if r["class3_acc"] != "" else 0 for r in pts]
        ax.plot(xs, leak, marker="o", color="#E45756", label="已见漏入")
        ax.plot(xs, c3, marker="s", color="#4C78A8", label="类 3 准确率")
        ax.set_ylim(-5, 105)
        ax.set_xticks(xs)
        ax.set_xlabel("会话")
        ax.set_ylabel("%")
        ax.set_title(dataset)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "10_registration_boundary.png")
    plt.close(fig)


def main() -> None:
    export_split()
    export_session_and_per_class()
    export_ablation()
    export_overlap()
    plot_lines()
    plot_bars()
    plot_ablation()
    plot_heat()
    plot_overlap()
    plot_registration()
    print("wrote tables to", OUT)
    print("wrote figures to", FIG)


if __name__ == "__main__":
    main()
