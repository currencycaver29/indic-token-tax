"""Charts + tables from results/ produced by measure.py."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from tokenizers_list import TOKENIZERS

NAMES = {"asm_Beng": "Assamese", "ben_Beng": "Bengali", "brx_Deva": "Bodo", "doi_Deva": "Dogri",
         "gom_Deva": "Konkani", "guj_Gujr": "Gujarati", "hin_Deva": "Hindi", "kan_Knda": "Kannada",
         "kas_Arab": "Kashmiri", "mai_Deva": "Maithili", "mal_Mlym": "Malayalam", "mar_Deva": "Marathi",
         "mni_Mtei": "Manipuri (Meitei)", "npi_Deva": "Nepali", "ory_Orya": "Odia", "pan_Guru": "Punjabi",
         "san_Deva": "Sanskrit", "sat_Olck": "Santali (Ol Chiki)", "snd_Deva": "Sindhi (Deva.)",
         "tam_Taml": "Tamil", "tel_Telu": "Telugu", "urd_Arab": "Urdu", "eng_Latn": "English"}
DAK = {"bn": "Bengali", "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi",
       "pa": "Punjabi", "sd": "Sindhi", "ta": "Tamil", "te": "Telugu", "ur": "Urdu"}
INDIAN = {"Sarvam-1", "IndicBERT v2", "MuRIL"}   # Krutrim-2 reuses Mistral NeMo's tokenizer, so it is not counted as Indian-built
ENCODERS = {"IndicBERT v2", "MuRIL", "XLM-R", "mBERT"}      # not used for chat / not billed per token
SARVAM10 = ["ben_Beng", "guj_Gujr", "hin_Deva", "kan_Knda", "mal_Mlym", "mar_Deva", "ory_Orya", "pan_Guru", "tam_Taml", "tel_Telu"]
SHARED = {"Mistral NeMo": "Mistral NeMo / Krutrim-2"}
UNK_LIMIT = 0.02          # cells where >2% of tokens are [UNK] are masked: they look cheap only because text is lost
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#8a8984", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})


def text_color(rgba):
    r, g, b = rgba[:3]
    return INK if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else "white"


def main():
    s = pd.read_csv("results/in22_summary.csv")
    info = pd.read_csv("results/tokenizer_info.csv")
    ps = pd.read_parquet("results/in22_per_sentence.parquet")
    d = pd.read_csv("results/dakshina_summary.csv")

    # identical tokenizers (same vocabulary fingerprint) are reported once
    dup = info[info.duplicated("fingerprint", keep="first")]
    dup_notes = [f"{r.tokenizer} has the same vocabulary as "
                 f"{info[(info.fingerprint == r.fingerprint)].tokenizer.iloc[0]}" for r in dup.itertuples()]
    keep = [t for t in info.tokenizer if t not in set(dup.tokenizer)]
    s, ps, d = s[s.tokenizer.isin(keep)], ps[ps.tokenizer.isin(keep)], d[d.tokenizer.isin(keep)]
    s, ps, d = (x.assign(tokenizer=x.tokenizer.replace(SHARED)) for x in (s, ps, d))
    info = info.assign(tokenizer=info.tokenizer.replace(SHARED))
    keep = [SHARED.get(t, t) for t in keep]

    indic = [l for l in s.lang.unique() if l != "eng_Latn"]
    prem = s.pivot(index="tokenizer", columns="lang", values="premium_vs_english")[indic]
    unk = s.pivot(index="tokenizer", columns="lang", values="unk_rate")[indic]
    valid = prem.where(unk <= UNK_LIMIT)

    # mean premium across the 22 languages, with a paired bootstrap 95% CI over sentences
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 1024, (1000, 1024))
    rows = []
    for t in keep:
        p = ps[ps.tokenizer == t].pivot(index="lang", columns="sent_id", values="tokens")
        ok = [l for l in indic if unk.loc[t, l] <= UNK_LIMIT]
        eng = p.loc["eng_Latn"].values[idx].sum(1)
        boots = np.mean([p.loc[l].values[idx].sum(1) / eng for l in ok], axis=0)
        worst = valid.loc[t].idxmax()
        rows.append({"tokenizer": t, "mean_premium": valid.loc[t].mean(),
                     "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
                     "best_lang": NAMES[valid.loc[t].idxmin()], "best": valid.loc[t].min(),
                     "worst_lang": NAMES[worst], "worst": valid.loc[t].max(),
                     "langs_masked_unk": int((unk.loc[t] > UNK_LIMIT).sum()),
                     "vocab_size": int(info.set_index("tokenizer").vocab_size[t]),
                     "origin": "Indian" if t in INDIAN else "Global",
                     "type": "encoder" if t in ENCODERS else "LLM",
                     "premium_sarvam10": valid.loc[t, SARVAM10].mean(),
                     "premium_other12": valid.loc[t, [l for l in indic if l not in SARVAM10]].mean()})
    m = pd.DataFrame(rows).sort_values("mean_premium")
    m.to_csv("results/mean_premium.csv", index=False)   # full precision; round only for display

    order_t = m.tokenizer.tolist()
    order_l = valid.mean().sort_values().index.tolist()
    prem.loc[order_t, order_l].rename(columns=NAMES).round(3).to_csv("results/premium_matrix.csv")

    # ---- Fig 1: heatmap -------------------------------------------------
    V = valid.loc[order_t, order_l]
    fig, ax = plt.subplots(figsize=(13, 6.2))
    vmax = np.nanpercentile(V.values, 98)
    im = ax.imshow(V.values, cmap=SEQ, vmin=1, vmax=vmax, aspect="auto")
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            v = V.values[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#e6e5e0"))
                ax.text(j, i, "UNK", ha="center", va="center", fontsize=6.5, color=INK2)
            else:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                        color=text_color(im.cmap(im.norm(v))))
    ax.set_xticks(range(len(order_l)), [NAMES[l] for l in order_l], rotation=50, ha="right")
    ax.set_yticks(range(len(order_t)), order_t)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in INDIAN: lbl.set_fontweight("bold")
    ax.set_xticks(np.arange(-.5, len(order_l)), minor=True); ax.set_yticks(np.arange(-.5, len(order_t)), minor=True)
    ax.grid(which="minor", color=SURF, linewidth=2); ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("tokens relative to English (×)")
    cb.outline.set_visible(False)
    ax.set_title("Token premium: how many times more tokens the same sentence needs than in English",
                 loc="left", fontsize=11, color=INK)
    fig.text(0.01, 0.01, "IN22-Gen, 1,024 parallel sentences. Bold = Indian-built tokenizers. "
             "UNK = over 2% of tokens are 'unknown' (text is dropped, so the count is meaningless).",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig("results/fig1_premium_heatmap.png", dpi=200); plt.close(fig)

    # ---- Fig 2: mean premium per tokenizer, LLMs and encoders in separate groups ----
    from matplotlib.patches import Patch
    llm = m[m.type == "LLM"].sort_values("mean_premium", ascending=False)
    enc = m[m.type == "encoder"].sort_values("mean_premium", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6.2))
    ypos, labels, y = [], [], 0
    for grp in (enc, llm):
        for r in grp.itertuples():
            ax.barh(y, r.mean_premium, color=ORANGE if r.origin == "Indian" else BLUE, height=0.7)
            ax.errorbar(r.mean_premium, y, xerr=[[r.mean_premium - r.ci_low], [r.ci_high - r.mean_premium]],
                        fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
            ax.text(r.ci_high + 0.08, y, f"{r.mean_premium:.2f}×", va="center", fontsize=8, color=INK2)
            labels.append(r.tokenizer + ("*" if r.langs_masked_unk else "")); ypos.append(y); y += 1
        y += 0.8
    ax.set_yticks(ypos, labels)
    ax.text(-0.30, (len(enc) - 1) / 2, "ENCODERS", transform=ax.get_yaxis_transform(), rotation=90,
            ha="right", va="center", fontsize=7.5, color=INK2)
    ax.text(-0.30, len(enc) + 0.8 + (len(llm) - 1) / 2, "LLMs (billed per token)", transform=ax.get_yaxis_transform(),
            rotation=90, ha="right", va="center", fontsize=7.5, color=INK2)
    ax.axvline(1, color="#8a8984", lw=1, ls="--")
    ax.set_xlabel("average token premium across 22 Indian languages (×)")
    ax.grid(axis="x", color="#e6e5e0"); ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=BLUE, label="Global"), Patch(color=ORANGE, label="Indian-built")],
              frameon=False, loc="lower right")
    ax.set_title("Average token premium by tokenizer (lower is better)", loc="left", fontsize=11, color=INK)
    fig.text(0.01, 0.01, "* average excludes languages where the tokenizer outputs over 2% [UNK] tokens "
             "(mBERT 7, XLM-R 3, MuRIL 2). Whiskers: 95% bootstrap CI. English = 1× (dashed).",
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0.04, 0.03, 1, 1)); fig.savefig("results/fig2_mean_premium.png", dpi=200); plt.close(fig)

    # ---- Fig 5: Sarvam's 10 target languages vs the other 12 ----------
    llm_s = m[m.type == "LLM"].sort_values("premium_sarvam10")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(llm_s)); w = 0.38
    b1 = ax.bar(x - w / 2, llm_s.premium_sarvam10, w * 0.92, color=BLUE, label="Sarvam-1's 10 target languages")
    b2 = ax.bar(x + w / 2, llm_s.premium_other12, w * 0.92, color=ORANGE, label="the other 12 scheduled languages")
    for bars in (b1, b2):
        for bb in bars:
            ax.annotate(f"{bb.get_height():.2f}", (bb.get_x() + bb.get_width() / 2, bb.get_height()), ha="center",
                        va="bottom", fontsize=7, color=INK2, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x, [t.replace(" / ", " /\n") for t in llm_s.tokenizer], fontsize=8, rotation=25, ha="right")
    ax.set_ylabel("average token premium (×)"); ax.margins(y=0.12)
    ax.grid(axis="y", color="#e6e5e0"); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=2, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_title("Built for some Indian languages is not built for all", loc="left", fontsize=11, color=INK, pad=24)
    fig.tight_layout(); fig.savefig("results/fig5_sarvam10_vs_other.png", dpi=200); plt.close(fig)

    # ---- Fig 3: GPT-4 -> GPT-4o ----------------------------------------
    if {"GPT-4", "GPT-4o"} <= set(keep):
        g = valid.loc[["GPT-4", "GPT-4o"]].T.dropna().sort_values("GPT-4o")
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        yy = range(len(g))
        ax.hlines(yy, g["GPT-4o"], g["GPT-4"], color="#c3c2b7", lw=2)
        ax.scatter(g["GPT-4"], yy, color="#8a8984", s=40, zorder=3, label="GPT-4 (cl100k)")
        ax.scatter(g["GPT-4o"], yy, color=BLUE, s=40, zorder=3, label="GPT-4o (o200k)")
        ax.set_yticks(yy, [NAMES[l] for l in g.index]); ax.set_xlabel("token premium vs English (×)")
        ax.grid(axis="x", color="#e6e5e0"); ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="lower right")
        ax.set_title("OpenAI's newer tokenizer cut the premium, but not evenly", loc="left", fontsize=11, color=INK)
        fig.tight_layout(); fig.savefig("results/fig3_gpt4_vs_gpt4o.png", dpi=200); plt.close(fig)

    # ---- Fig 4: romanized vs native ------------------------------------
    r = d.pivot(index="tokenizer", columns="lang_code", values="roman_over_native")
    r = r.loc[order_t]
    r = r[r.mean().sort_values().index]
    fig, ax = plt.subplots(figsize=(9, 6))
    lo, hi = np.nanmin(r.values), np.nanmax(r.values)
    norm = TwoSlopeNorm(vmin=min(lo, 0.99), vcenter=1.0, vmax=max(hi, 1.01))
    im = ax.imshow(r.values, cmap=DIV, norm=norm, aspect="auto")
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            v = r.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=text_color(im.cmap(norm(v))))
    ax.set_xticks(range(r.shape[1]), [DAK[c] for c in r.columns], rotation=40, ha="right")
    ax.set_yticks(range(r.shape[0]), r.index)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in INDIAN: lbl.set_fontweight("bold")
    ax.set_xticks(np.arange(-.5, r.shape[1]), minor=True); ax.set_yticks(np.arange(-.5, r.shape[0]), minor=True)
    ax.grid(which="minor", color=SURF, linewidth=2); ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01); cb.outline.set_visible(False)
    cb.set_label("romanized tokens ÷ native-script tokens")
    ax.set_title("Typing in English letters: below 1 (blue) = romanized is cheaper", loc="left", fontsize=11, color=INK)
    fig.text(0.01, 0.01, "Dakshina dataset: the same sentences in native script and human-written romanization. "
             "Note: tokenizers that turn native script into [UNK] look artificially cheap in native script.",
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig("results/fig4_romanized.png", dpi=200); plt.close(fig)
    r.rename(columns=DAK).round(3).to_csv("results/romanized_ratio_matrix.csv")

    # ---- key numbers for the write-up ----------------------------------
    with open("results/key_numbers.md", "w") as f:
        f.write("## Mean premium per tokenizer (22 languages, 95% bootstrap CI)\n\n")
        f.write(m.round(2).to_markdown(index=False) + "\n\n")
        f.write("## Per-language premium (mean across tokenizers, UNK-masked)\n\n")
        f.write(valid.mean().sort_values().rename(index=NAMES).round(2).to_markdown() + "\n\n")
        f.write("## Romanized / native token ratio (mean across tokenizers)\n\n")
        f.write(r.mean().rename(index=DAK).round(2).to_markdown() + "\n\n")
        f.write("## Romanized / native per tokenizer (mean across languages)\n\n")
        f.write(r.mean(axis=1).round(2).to_markdown() + "\n\n")
        f.write("## UNK rates above 2%\n\n")
        u = s[s.unk_rate > UNK_LIMIT][["tokenizer", "lang", "unk_rate"]]
        f.write((u.assign(lang=u.lang.map(NAMES)).round(3).to_markdown(index=False) if len(u) else "none") + "\n\n")
        f.write("## Byte-fallback rates above 5%\n\n")
        b = s[s.byte_fallback_rate > 0.05][["tokenizer", "lang", "byte_fallback_rate"]]
        f.write((b.assign(lang=b.lang.map(NAMES)).round(3).to_markdown(index=False) if len(b) else "none") + "\n\n")
        f.write("## Premium on Sarvam-1's 10 languages vs other 12\n\n")
        f.write(m[["tokenizer", "type", "premium_sarvam10", "premium_other12"]].round(2).to_markdown(index=False) + "\n\n")
        nf = pd.read_csv("results/nfc_effect.csv")
        f.write("## NFC normalization effects beyond ±1%\n\n")
        f.write(nf[nf.change.abs() > 0.01].assign(lang=lambda x: x.lang.map(NAMES)).sort_values("change")
                [["tokenizer", "lang", "change"]].round(3).to_markdown(index=False) + "\n\n")
        f.write("## Duplicates removed\n\n" + ("\n".join(dup_notes) or "none") + "\n")
    print(open("results/key_numbers.md").read())


if __name__ == "__main__":
    main()
