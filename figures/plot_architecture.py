"""Overall architecture of one unlabeled session."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(r"e:\WPSCloud\iepe\workspace\papers\figures")
font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    }
)

C_IN = "#F7F7F7"
C_OOD = "#D9E8F6"
C_NCD = "#D9F0E4"
C_CIL = "#FDE8D0"
C_DEC = "#FFF6D8"
C_EDGE = "#3A3A3A"


def box(ax, x, y, w, h, text, fc, fontsize=9.5):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.18",
        facecolor=fc,
        edgecolor=C_EDGE,
        linewidth=0.8,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=0.8,
            color="#333333",
            shrinkA=2,
            shrinkB=2,
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(11.6, 6.4))
    ax.set_xlim(0, 118)
    ax.set_ylim(0, 64)
    ax.axis("off")
    ax.set_aspect("equal")

    # stage bands
    for x, w, title, color in (
        (2, 36, "OOD  残差", "#EEF5FB"),
        (40, 36, "NCD  登记", "#EEF8F2"),
        (78, 36, "CIL  记忆更新", "#FFF6EC"),
    ):
        band = FancyBboxPatch(
            (x, 3),
            w,
            54,
            boxstyle="round,pad=0.2,rounding_size=0.4",
            facecolor=color,
            edgecolor="none",
            zorder=0,
        )
        ax.add_patch(band)
        ax.text(x + w / 2, 59.2, title, ha="center", va="center", fontsize=12, color="#222222")

    # OOD
    box(ax, 20, 50, 28, 7.2, r"无标签到达包" + "\n" + r"$\mathcal{X}_t$", C_IN)
    box(ax, 20, 39, 28, 7.2, r"记忆上的对角高斯" + "\n" + r"$\{\pi_c,\mu_c,v_c\}$", C_OOD)
    box(ax, 20, 28, 28, 7.2, r"马氏残差 $r(x)$" + "\n" + r"与 $\chi^2_{d,0.95}$", C_OOD)
    box(ax, 20, 17, 28, 7.2, r"解释不了的行" + "\n" + r"$\mathcal{U}_t$", C_OOD)
    arrow(ax, 20, 46.4, 20, 42.6)
    arrow(ax, 20, 35.4, 20, 31.6)
    arrow(ax, 20, 24.4, 20, 20.6)
    ax.text(8, 43.5, r"$\mathcal{E}_{t-1}$", fontsize=9, color="#333333")
    arrow(ax, 12.2, 43.2, 16.2, 41.2)

    # NCD
    box(ax, 58, 50, 28, 7.2, r"轮廓系数估计" + "\n" + r"$\hat{K}_t=\mathrm{argmax}_k\, s(k)$", C_NCD)
    box(ax, 58, 32, 28, 8.6, r"$\mathcal{U}_t$ 上 $k$ 均值初值" + "\n" + r"整包 30 步锚定 EM", C_NCD)
    box(ax, 58, 14, 28, 8.2, r"最大责任度" + "\n" + r"旧类保持；新簇 $\geq 5$", C_NCD)
    arrow(ax, 34, 17, 44, 50)
    arrow(ax, 58, 46.4, 58, 36.3)
    arrow(ax, 58, 27.7, 58, 18.1)
    ax.text(60.2, 41.2, r"$\hat{K}_t>0$", fontsize=8, color="#444444")

    # CIL
    box(ax, 96, 50, 26, 7.2, r"$\hat{K}_t=0$：不登记" + "\n" + r"保持 $f_{t-1},\mathcal{E}_{t-1}$", C_DEC, fontsize=9)
    box(ax, 96, 32, 26, 8.4, r"$\mathcal{L}_{\mathrm{SupCon}}+\mathcal{L}_{\mathrm{distill}}$" + "\n" + r"冻结 $f_{t-1}$，对比只在新类", C_CIL, fontsize=9)
    box(ax, 96, 14, 26, 8.6, r"MES herding 与 BRF" + "\n" + r"$m_t=\lceil M/|\mathcal{C}_t|\rceil$", C_CIL)
    arrow(ax, 72, 50, 83, 50)
    arrow(ax, 72, 14, 83, 32)
    arrow(ax, 96, 27.8, 96, 18.3)
    ax.annotate(
        "",
        xy=(112.2, 18.3),
        xytext=(112.2, 46.4),
        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.8, mutation_scale=11),
    )
    ax.plot([109, 112.2], [46.4, 46.4], color="#333333", lw=0.8)
    ax.text(113.0, 32, r"保持", fontsize=8, color="#444444")

    fig.tight_layout(pad=0.3)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "04_architecture.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
