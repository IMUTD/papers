"""Visualize the three open-world problems on TE and MFF session files.

Coordinates are the exported vectors in [0, 1]. Labels are used only to color
points and to count rows. They are not written back into any training step.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(r"e:\WPSCloud\iepe\workspace")
OUT = ROOT / "papers" / "figures"
DATA = {
    "TE": ROOT / "sysbaseline" / "te_sessions.npz",
    "MFF": ROOT / "sysbaseline" / "mff_sessions.npz",
}
REG = 1e-4
SEED = 66
MIN_CLUSTER = 5

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


def chi2_limit(dim: int, q: float = 0.95) -> float:
    from scipy.stats import chi2

    return float(chi2.ppf(q, dim))


def load(name: str):
    z = np.load(DATA[name], allow_pickle=True)
    sessions = []
    t = 0
    while f"s{t}_x" in z.files:
        sessions.append(
            {
                "x": z[f"s{t}_x"].astype(np.float64),
                "y": z[f"s{t}_y"].astype(np.int64),
                "test_x": z[f"s{t}_test_x"].astype(np.float64),
                "test_y": z[f"s{t}_test_y"].astype(np.int64),
                "seen": z[f"s{t}_seen"].astype(np.int64),
            }
        )
        t += 1
    return sessions


def class_gaussian(x: np.ndarray, y: np.ndarray, classes: np.ndarray):
    mu, var = {}, {}
    for c in classes:
        xc = x[y == c]
        mu[int(c)] = xc.mean(axis=0)
        var[int(c)] = xc.var(axis=0) + REG
    return mu, var


def nearest_maha(x: np.ndarray, mu: dict, var: dict, classes: np.ndarray):
    dists = []
    labels = []
    for c in classes:
        c = int(c)
        d = ((x - mu[c]) ** 2 / var[c]).sum(axis=1)
        dists.append(d)
        labels.append(c)
    stacked = np.vstack(dists)
    idx = stacked.argmin(axis=0)
    return stacked.min(axis=0), np.asarray(labels, dtype=np.int64)[idx]


def pca_xy(x: np.ndarray):
    pca = PCA(n_components=2, random_state=SEED)
    xy = pca.fit_transform(x)
    return xy, pca.explained_variance_ratio_


def style_ax(ax, title: str, ratio):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(f"PC1  {ratio[0] * 100:.1f}%")
    ax.set_ylabel(f"PC2  {ratio[1] * 100:.1f}%")
    ax.tick_params(labelsize=8)


def plot_mix(bundles: dict):
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.2))
    for row, name in enumerate(["TE", "MFF"]):
        s1 = bundles[name]["s1"]
        xy, ratio = s1["pca"]
        seen = s1["is_seen"]
        ax = axes[row, 0]
        ax.scatter(xy[~seen, 0], xy[~seen, 1], s=26, c="#4C78A8", alpha=0.8, label="新异常", linewidths=0)
        ax.scatter(xy[seen, 0], xy[seen, 1], s=42, c="#F58518", alpha=0.95, label="已见再现", linewidths=0, zorder=3)
        style_ax(ax, f"{name}  第 1 轮到达包", ratio)
        ax.legend(frameon=False, fontsize=8)
        ax.text(
            0.02,
            0.98,
            f"已见 {seen.sum()} 行 / 共 {len(seen)} 行",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )

        ax = axes[row, 1]
        y = s1["y"]
        colors = plt.cm.tab10(np.linspace(0, 0.9, 10))
        for c in np.unique(y):
            m = y == c
            kind = "再现" if c in s1["seen_set"] else "新"
            ax.scatter(
                xy[m, 0],
                xy[m, 1],
                s=36,
                color=colors[int(c) % 10],
                alpha=0.9,
                label=f"类 {int(c)}（{kind}，{m.sum()}）",
                linewidths=0,
            )
        style_ax(ax, f"{name}  按真实类别着色（只用于显示）", ratio)
        ax.legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("问题 1  同一到达包里既有已见再现，也有新异常；前两主成分上可以分开，但包内没有标签", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "01_mixed_arrival.png")
    plt.close(fig)


def plot_discovery(bundles: dict):
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.2))
    for row, name in enumerate(["TE", "MFF"]):
        s1 = bundles[name]["s1"]
        xy, ratio = s1["pca"]
        unexplained = s1["unexplained"]
        y = s1["y"]
        ax = axes[row, 0]
        ax.scatter(xy[~unexplained, 0], xy[~unexplained, 1], s=22, c="#C7C7C7", label="残差以内", linewidths=0)
        if unexplained.any():
            km = KMeans(n_clusters=2, n_init=10, random_state=SEED)
            lab = km.fit_predict(s1["x"][unexplained])
            pts = xy[unexplained]
            yy = y[unexplained]
            for k, color in enumerate(["#4C78A8", "#54A24B"]):
                m = lab == k
                ax.scatter(pts[m, 0], pts[m, 1], s=36, c=color, label=f"新簇 {k}（{m.sum()} 行）", linewidths=0)
                counts = {int(c): int((yy[m] == c).sum()) for c in np.unique(yy[m])}
                s1.setdefault("cluster_counts", {})[k] = counts
            seen_out = int((s1["is_seen"] & unexplained).sum())
            s1["seen_unexplained"] = seen_out
        style_ax(ax, f"{name}  残差超过卡方 0.95 的行再分成 2 簇", ratio)
        ax.legend(frameon=False, fontsize=7)
        limit = s1["limit"]
        n_out = int(unexplained.sum())
        ax.text(
            0.02,
            0.98,
            f"界限 {limit:.1f}    门外 {n_out} / {len(y)}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )

        ax = axes[row, 1]
        classes = np.unique(y)
        out_counts = [int((y[unexplained] == c).sum()) for c in classes]
        totals = [int((y == c).sum()) for c in classes]
        bar_colors = ["#F58518" if c in s1["seen_set"] else "#4C78A8" for c in classes]
        bars = ax.bar([f"类 {int(c)}" for c in classes], out_counts, color=bar_colors)
        ax.axhline(MIN_CLUSTER, color="#E45756", ls="--", lw=1, label=f"最少 {MIN_CLUSTER} 行")
        for bar, out_n, total in zip(bars, out_counts, totals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"门外 {out_n}\n到达 {total}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.set_ylabel("门外行数")
        ax.set_ylim(0, max(out_counts + [MIN_CLUSTER]) * 1.28)
        ax.set_title(f"{name}  到达行数与门外行数")
        ax.legend(frameon=False, fontsize=8)
        ax.tick_params(labelsize=8)
    fig.suptitle("问题 2  TE 的类 3 大多留在旧类高斯内；MFF 的类 2、类 3 整团在门外", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "02_unexplained_clusters.png")
    plt.close(fig)


def plot_overlap(bundles: dict):
    spec = {
        "TE": {"session": 4, "focus": [0, 8], "title": "TE  最后一轮测试"},
        "MFF": {"session": 2, "focus": [0, 4], "title": "MFF  最后一轮测试"},
    }
    colors = {0: "#F58518", 4: "#4C78A8", 8: "#E45756"}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    for ax, name in zip(axes, ["TE", "MFF"]):
        conf = spec[name]
        sess = bundles[name]["sessions"][conf["session"]]
        xy, ratio = pca_xy(sess["test_x"])
        y = sess["test_y"]
        focus = set(conf["focus"])
        other = ~np.isin(y, list(focus))
        ax.scatter(xy[other, 0], xy[other, 1], s=8, c="#D0D0D0", label="其余已到达类", linewidths=0, zorder=1)
        for c in conf["focus"]:
            m = y == c
            ax.scatter(
                xy[m, 0],
                xy[m, 1],
                s=18 if c == conf["focus"][0] else 12,
                facecolors="none" if c == conf["focus"][0] else colors[c],
                edgecolors=colors[c],
                linewidths=0.6,
                alpha=0.85,
                label=f"类 {c}（{m.sum()}）",
                zorder=3 if c == conf["focus"][0] else 2,
            )
        style_ax(ax, conf["title"], ratio)
        ax.legend(frameon=False, fontsize=8, markerscale=1.4)
    fig.suptitle("问题 3  TE 上类 0 与类 8 落在同一团；MFF 上类 0 与类 4 分开", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "03_late_class_overlap.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bundles = {}
    for name in ["TE", "MFF"]:
        sessions = load(name)
        s0, s1 = sessions[0], sessions[1]
        classes = np.unique(s0["y"])
        mu, var = class_gaussian(s0["x"], s0["y"], classes)
        dist, _ = nearest_maha(s1["x"], mu, var, classes)
        limit = chi2_limit(s1["x"].shape[1])
        seen_set = set(int(c) for c in s1["seen"])
        is_seen = np.isin(s1["y"], list(seen_set))
        xy, ratio = pca_xy(s1["x"])
        bundles[name] = {
            "sessions": sessions,
            "s1": {
                "x": s1["x"],
                "y": s1["y"],
                "seen_set": seen_set,
                "is_seen": is_seen,
                "pca": (xy, ratio),
                "unexplained": dist > limit,
                "limit": limit,
                "dist": dist,
            },
        }
        out = dist > limit
        seen_out = int((is_seen & out).sum())
        seen_n = int(is_seen.sum())
        print(
            f"{name} s1 seen {seen_n} unexplained_seen {seen_out} "
            f"rate {seen_out / seen_n:.3f} unexplained {int(out.sum())}/{len(out)} limit {limit:.2f}"
        )
        y = s1["y"]
        for c in np.unique(y):
            print(f"  class {int(c)} total {(y == c).sum()} unexplained {(out & (y == c)).sum()}")

    plot_mix(bundles)
    plot_discovery(bundles)
    plot_overlap(bundles)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
