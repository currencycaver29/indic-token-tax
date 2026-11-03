"""Effect of Unicode NFC normalization on token counts (same text, renders identically).
Output: results/nfc_effect.csv  (tokenizer, lang, tokens_raw, tokens_nfc, change)"""
import unicodedata, warnings, pandas as pd
warnings.filterwarnings("ignore")
from measure import load, encode
from tokenizers_list import TOKENIZERS

in22 = pd.read_parquet("data/in22_gen_test.parquet")
changed = {l: (in22[l] != in22[l].map(lambda x: unicodedata.normalize("NFC", x))).mean() for l in in22.columns}
rows = []
for label, repo, _ in TOKENIZERS:
    if label == "Krutrim-2":
        continue                                  # identical to Mistral NeMo
    tok = load(repo)
    for lang in in22.columns:
        if changed[lang] == 0:
            continue
        a = sum(map(len, encode(tok, in22[lang])))
        b = sum(map(len, encode(tok, in22[lang].map(lambda x: unicodedata.normalize("NFC", x)))))
        rows.append({"tokenizer": label, "lang": lang, "sentences_changed_by_nfc": round(changed[lang], 3),
                     "tokens_raw": a, "tokens_nfc": b, "change": (b - a) / a})
    print(label, "done", flush=True)
r = pd.DataFrame(rows)
r.to_csv("results/nfc_effect.csv", index=False)
print("languages whose text NFC changes:", {k: round(v, 3) for k, v in changed.items() if v})
print(r[r.change.abs() > 0.01].sort_values("change").round(3).to_string(index=False))
