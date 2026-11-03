"""IEEE-sized vector figures (3.5in single column, 7.16in double column) from results/."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "results"); O = os.path.join(H, "figs"); os.makedirs(O, exist_ok=True)
import sys; sys.path.insert(0, os.path.join(H, "..")); from analyze import NAMES, DAK, INDIAN
INK, INK2, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.edgecolor": "#8a8984", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
tc = lambda rgba: INK if 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] > 0.55 else "white"

s = pd.read_csv(f"{R}/in22_summary.csv"); m = pd.read_csv(f"{R}/mean_premium.csv")
s = s[s.tokenizer != "Krutrim-2"].assign(tokenizer=lambda x: x.tokenizer.replace({"Mistral NeMo": "Mistral NeMo / Krutrim-2"}))
indic = [l for l in s.lang.unique() if l != "eng_Latn"]
P = s.pivot(index="tokenizer", columns="lang", values="premium_vs_english")[indic]
U = s.pivot(index="tokenizer", columns="lang", values="unk_rate")[indic]
V = P.where(U <= 0.02)
order_t = m.sort_values("mean_premium").tokenizer.tolist(); order_l = V.mean().sort_values().index.tolist()

def heat(ax, M, cmap, norm, fmt, rows, cols, masked_text="UNK", fs=5):
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#e6e5e0")); ax.text(j, i, masked_text, ha="center", va="center", fontsize=fs - 0.5, color=INK2)
            else:
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs, color=tc(im.cmap(im.norm(v))))
    ax.set_xticks(range(len(cols)), cols, rotation=55, ha="right"); ax.set_yticks(range(len(rows)), rows)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in INDIAN: lbl.set_fontweight("bold")
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True); ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1); ax.tick_params(which="minor", length=0); ax.tick_params(length=2)
    for sp in ax.spines.values(): sp.set_visible(False)
    return im

# Fig 1 (double column): premium heatmap
fig, ax = plt.subplots(figsize=(7.16, 3.0))
Mv = V.loc[order_t, order_l].values
im = heat(ax, Mv, SEQ, plt.Normalize(1, np.nanpercentile(Mv, 98)), "{:.1f}", order_t, [NAMES[l] for l in order_l])
cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.outline.set_visible(False); cb.set_label("premium vs. English (×)", fontsize=6); cb.ax.tick_params(labelsize=6)
fig.savefig(f"{O}/fig1_heatmap.pdf"); plt.close(fig)

# Fig 2 (single column): GPT-4 vs GPT-4o
g = V.loc[["GPT-4", "GPT-4o"]].T.sort_values("GPT-4o")
fig, ax = plt.subplots(figsize=(3.5, 3.3))
yy = np.arange(len(g))
ax.hlines(yy, g["GPT-4o"], g["GPT-4"], color="#c3c2b7", lw=1.5)
ax.scatter(g["GPT-4"], yy, color="#8a8984", s=14, zorder=3, label="GPT-4 (cl100k\\_base)".replace("\\_", "_"))
ax.scatter(g["GPT-4o"], yy, color=BLUE, s=14, zorder=3, label="GPT-4o (o200k_base)")
ax.set_yticks(yy, [NAMES[l] for l in g.index]); ax.set_xlabel("premium vs. English (×)")
ax.grid(axis="x", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, loc="lower right", fontsize=6)
fig.savefig(f"{O}/fig2_gpt4_vs_gpt4o.pdf"); plt.close(fig)

# Fig 3 (single column): Sarvam-10 vs other-12 for LLM tokenizers
l = m[m.type == "LLM"].sort_values("premium_sarvam10")
fig, ax = plt.subplots(figsize=(3.5, 2.3))
x = np.arange(len(l)); w = 0.38
ax.bar(x - w / 2, l.premium_sarvam10, w * 0.9, color=BLUE, label="Sarvam-1's 10 target languages")
ax.bar(x + w / 2, l.premium_other12, w * 0.9, color=ORANGE, label="other 12 scheduled languages")
ax.set_xticks(x, [t.replace("Mistral NeMo / Krutrim-2", "Mistral NeMo/\nKrutrim-2") for t in l.tokenizer], rotation=40, ha="right", fontsize=6)
ax.set_ylabel("mean premium (×)"); ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=6, loc="upper left")
fig.savefig(f"{O}/fig3_sarvam_split.pdf"); plt.close(fig)

# Fig 4 (single column): romanized / native ratio
r = pd.read_csv(f"{R}/romanized_ratio_matrix.csv").set_index("tokenizer").loc[order_t]
r = r[r.mean().sort_values().index]
fig, ax = plt.subplots(figsize=(3.5, 3.0))
im = heat(ax, r.values, DIV, TwoSlopeNorm(vmin=min(np.nanmin(r.values), 0.99), vcenter=1.0, vmax=max(np.nanmax(r.values), 1.01)), "{:.2f}", r.index.tolist(), r.columns.tolist(), fs=4.5)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.outline.set_visible(False); cb.set_label("romanized ÷ native tokens", fontsize=6); cb.ax.tick_params(labelsize=6)
fig.savefig(f"{O}/fig4_romanized.pdf"); plt.close(fig)
print("figures:", sorted(os.listdir(O)))
