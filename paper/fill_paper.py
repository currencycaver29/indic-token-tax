"""Fills main_template.tex with numbers from results/ -> main.tex. Author details from author.json if present."""
import json, os, runpy
import pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "results")
num = runpy.run_path(os.path.join(H, "..", "article", "fill_article.py"))["num"]   # same numbers as the article
import sys; sys.path.insert(0, os.path.join(H, ".."))
from importlib import util
spec = util.spec_from_file_location("an", os.path.join(H, "..", "analyze.py")); an = util.module_from_spec(spec); spec.loader.exec_module(an)
NAMES = an.NAMES
m = pd.read_csv(f"{R}/mean_premium.csv")
DEV = {"GPT-4o": ("OpenAI", "gpt4o"), "GPT-4": ("OpenAI", "gpt4"), "Llama 3": ("Meta", "llama3"), "Llama 4": ("Meta", "llama4"),
       "Gemma 3": ("Google", "gemma3"), "Mistral NeMo / Krutrim-2": ("Mistral AI / Ola", "mistralnemo,krutrim2"),
       "Qwen 3": ("Alibaba", "qwen3"), "DeepSeek-V3": ("DeepSeek", "deepseekv3"), "Sarvam-1": ("Sarvam AI", "sarvam1"),
       "BLOOM": ("BigScience", "bloom2022"), "IndicBERT v2": ("AI4Bharat", "doddapaneni2023"), "MuRIL": ("Google", "khanuja2021"),
       "mBERT": ("Google", "devlin2019"), "XLM-R": ("Meta", "conneau2020")}
def row(r):
    dev, cite = DEV[r.tokenizer]; star = "$^*$" if r.langs_masked_unk else ""
    name = r.tokenizer
    return (f"{name}{star}~\\cite{{{cite}}} & {dev} & {r.vocab_size / 1000:.0f}k & "
            f"{r.mean_premium:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}] & {r.best_lang} ({r.best:.2f}) & {r.worst_lang} ({r.worst:.1f}) \\\\")
llm = m[m.type == "LLM"].sort_values("mean_premium"); enc = m[m.type == "encoder"].sort_values("mean_premium")
nf = pd.read_csv(f"{R}/nfc_effect.csv").assign(a=lambda x: x.change.abs()).sort_values("a", ascending=False).head(8)
nf_rows = "\n".join(f"{t.replace('Mistral NeMo', 'Mistral NeMo/Krutrim-2')} & {NAMES[l]} & {'$-$' if c < 0 else '+'}{abs(c) * 100:.1f}\\% \\\\"
                    for t, l, c in zip(nf.tokenizer, nf.lang, nf.change))
rr = pd.read_csv(f"{R}/romanized_ratio_matrix.csv").set_index("tokenizer").mean(axis=1)[["Sarvam-1", "MuRIL", "IndicBERT v2"]]
author = json.load(open(f"{H}/author.json")) if os.path.exists(f"{H}/author.json") else {}
fill = {**num,
        "TABLE_LLM": "\n".join(row(r) for r in llm.itertuples()), "TABLE_ENC": "\n".join(row(r) for r in enc.itertuples()),
        "TABLE_NFC": nf_rows, "r_indic_range": f"{(rr.min() - 1) * 100:.0f}--{(rr.max() - 1) * 100:.0f}\\%",
        "AUTHOR": author.get("name", "Author Name"),
        "EMAIL_LINE": f"\\\\ {author['email']}" if author.get("email") else "",
        "REPO_URL": (f"at \\url{{{author['repo']}}}" if author.get("repo") else
                     "in the Zenodo record of this preprint")}
t = open(f"{H}/main_template.tex").read()
for k, v in fill.items(): t = t.replace(f"<<{k}>>", str(v))
assert "<<" not in t, [l for l in t.splitlines() if "<<" in l][:3]
open(f"{H}/main.tex", "w").write(t); print("main.tex written")
