"""生成实验报告所需的可视化图表（学术论文风格，无内嵌标题）"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# ---------- 学术风格全局设置 ----------
plt.rcParams.update({
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "sans-serif"],
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 250,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.alpha": 0.15,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "image")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 学术配色 ----------
# 蓝色系主色 + 暖色强调，适合黑白打印也能区分
C_BASELINE  = "#A0A0A0"   # gray — baseline
C_GRAPH     = "#E8A87C"   # muted orange
C_VECTOR    = "#6C9BD2"   # muted blue
C_NOIMG     = "#8DB27C"   # muted green
C_FULL      = "#D97471"   # muted red
GROUP_COLORS = [C_BASELINE, C_GRAPH, C_VECTOR, C_NOIMG, C_FULL]

GROUPS = ["none", "graph_only", "vector_only", "no_image", "full"]
GROUP_LABELS = ["无检索\n基线", "纯图谱\n检索", "纯向量\n检索", "无页面\n图像", "完整\n系统"]

# ---------- 数据 ----------
metrics_data = {
    "ANLS":       [0.0088, 0.0044, 0.0549, 0.0790, 0.0840],
    "Token F1":   [0.1924, 0.4007, 0.5704, 0.6074, 0.6185],
    "Heuristic\nJudge": [0.1042, 0.3042, 0.4408, 0.4750, 0.4813],
    "Evidence\nPage Recall":[0.0000, 0.8917, 0.8750, 0.9167, 0.9250],
}

qtype_labels = ["文本定位", "表格问答", "图表理解", "跨页延续", "多跳关系", "引用追踪"]
qtype_anls   = [0.1535, 0.1218, 0.0989, 0.0232, 0.0000, 0.0000]
qtype_tokf1  = [0.6305, 0.6461, 0.6102, 0.6018, 0.5898, 0.5976]
qtype_heu    = [0.4767, 0.5019, 0.4765, 0.4795, 0.4792, 0.4583]
qtype_counts = [30, 27, 17, 22, 12, 12]


def _add_value_labels(ax, bars, values, fmt=".4f", offset_ratio=0.02, fontsize=7.5):
    """在柱状图上方添加数值标注"""
    ymax = max(values)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * offset_ratio,
                f"{v:{fmt}}", ha="center", va="bottom", fontsize=fontsize, color="#333333")


# ======================== 图1：消融实验四指标对比 ========================
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

for ax, (metric_name, values) in zip(axes.flat, metrics_data.items()):
    bars = ax.bar(GROUP_LABELS, values, color=GROUP_COLORS, edgecolor="white", linewidth=0.3, width=0.65)
    display_name = metric_name.replace("\n", " ")
    ax.set_ylabel(display_name, fontsize=10)
    ymax = max(values)
    ax.set_ylim(0, ymax * 1.18)
    _add_value_labels(ax, bars, values, fmt=".4f" if ymax < 1 else ".3f", offset_ratio=0.03)
    ax.grid(axis="y")
    ax.tick_params(axis="x", labelsize=8.5)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "ablation_metrics.png"), facecolor="white")
plt.close(fig)


# ======================== 图2：按问题类型 ANLS + Token F1 双轴柱状 ========================
fig, ax1 = plt.subplots(figsize=(9.5, 5))

x = np.arange(len(qtype_labels))
width = 0.33

bars1 = ax1.bar(x - width / 2, qtype_anls, width, label="ANLS",
                color=C_VECTOR, edgecolor="white", linewidth=0.3)
ax1.set_ylabel("ANLS", color=C_VECTOR, fontsize=10)
ax1.set_ylim(0, max(qtype_anls) * 1.45)
ax1.tick_params(axis="y", labelcolor=C_VECTOR)

ax2 = ax1.twinx()
bars2 = ax2.bar(x + width / 2, qtype_tokf1, width, label="Token F1",
                color=C_FULL, edgecolor="white", linewidth=0.3)
ax2.set_ylabel("Token F1", color=C_FULL, fontsize=10)
ax2.set_ylim(0, 0.85)
ax2.tick_params(axis="y", labelcolor=C_FULL)

for bar, v in zip(bars1, qtype_anls):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
             f"{v:.4f}", ha="center", va="bottom", fontsize=7, color="#333333")
for bar, v in zip(bars2, qtype_tokf1):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
             f"{v:.4f}", ha="center", va="bottom", fontsize=7, color="#333333")

ax1.set_xticks(x)
ax1.set_xticklabels(qtype_labels, fontsize=9)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True,
           fancybox=False, edgecolor="#cccccc", framealpha=1.0)

ax1.grid(axis="y")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "by_question_type.png"), facecolor="white")
plt.close(fig)


# ======================== 图3：组件增量瀑布图 ========================
fig, ax = plt.subplots(figsize=(8, 4.8))

stages = ["无检索基线", "+ FAISS\n向量检索", "+ 图谱\n邻域扩展", "+ 页面原图\n(完整系统)"]
anls_cumulative = [0.0088, 0.0549, 0.0790, 0.0840]
deltas = [0.0088, 0.0461, 0.0241, 0.0050]

waterfall_colors = [C_BASELINE, C_VECTOR, C_NOIMG, C_FULL]
bars = ax.bar(stages, anls_cumulative, color=waterfall_colors, edgecolor="white",
              linewidth=0.5, width=0.55)

for i, (bar, cum, delta) in enumerate(zip(bars, anls_cumulative, deltas)):
    # 累计值标注在柱顶
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0025,
            f"{cum:.4f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#222222")
    # 增量标注在柱内（跳过第一个基线）
    if i > 0:
        mid_y = anls_cumulative[i - 1] + delta / 2
        ax.text(bar.get_x() + bar.get_width() / 2, mid_y,
                f"+{delta:.4f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

ax.set_ylabel("ANLS", fontsize=10)
ax.set_ylim(0, 0.105)
ax.grid(axis="y")
ax.tick_params(axis="x", labelsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "waterfall.png"), facecolor="white")
plt.close(fig)


# ======================== 图4：Token 节省率饼图 ========================
fig, ax = plt.subplots(figsize=(6, 5))

sizes = [6, 94]
labels = ["DocRAG 输入\n(约 5 页，6%)", "节省 token\n(94%)"]
pie_colors = ["#4A7FB5", "#E2E8F0"]
explode = (0.03, 0)

wedges, texts, autotexts = ax.pie(
    sizes, explode=explode, labels=labels, colors=pie_colors,
    autopct="%1.0f%%", startangle=90, pctdistance=0.58,
    textprops={"fontsize": 10, "color": "#333333"},
    wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
)
for at in autotexts:
    at.set_fontweight("bold")
    at.set_fontsize(11)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "token_savings.png"), facecolor="white")
plt.close(fig)

print(f"Charts saved to {OUTPUT_DIR}")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith((".png",)) and f != "DocRAG架构.png" and "SYSU" not in f and "sysu" not in f:
        print(f"  {f}")
