"""Counts tokens for every tokenizer x language.

Outputs (results/):
  in22_per_sentence.parquet   token count per sentence, tokenizer, language (IN22-Gen, 1024 parallel sentences)
  in22_summary.csv            per tokenizer x language: tokens, words, chars, fertility, premium vs English, UNK/byte-fallback rates
  dakshina_summary.csv        per tokenizer x language: tokens for native script vs human romanization (Dakshina)
  tokenizer_info.csv          vocab size + fingerprint (to spot tokenizers that are identical)
"""
import os, re, hashlib, warnings
import numpy as np, pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers_list import TOKENIZERS

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)
BYTE_TOKEN = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


TIKTOKEN = {"Xenova/gpt-4o": "o200k_base", "Xenova/gpt-4": "cl100k_base"}


class Tok:
    """Uniform wrapper. Loading order, chosen after cross-checking (see README):
    - GPT models: OpenAI's official tiktoken encodings (ground truth).
    - Others with tokenizer.json: the raw `tokenizers` file (identical to transformers for these).
    - MuRIL (vocab.txt only): transformers BertTokenizerFast.
    Note: transformers' AutoTokenizer mis-tokenized the GPT tokenizer.json conversions
    (GPT-4o +46% tokens vs official tiktoken), so it is NOT used for them."""
    def __init__(self, repo):
        d = f"tok/{repo.replace('/', '__')}"
        self.kind, self.unk_id, self.byte_ids = None, None, set()
        if repo in TIKTOKEN:
            import tiktoken
            self.enc = tiktoken.get_encoding(TIKTOKEN[repo]); self.kind = "tiktoken"
            self.vocab = {str(i): i for i in range(self.enc.n_vocab)}
            self.fingerprint_src = TIKTOKEN[repo]
            return
        hf = AutoTokenizer.from_pretrained(d)          # only used for metadata (UNK) / MuRIL
        self.unk_id = hf.unk_token_id
        if os.path.exists(os.path.join(d, "tokenizer.json")):
            from tokenizers import Tokenizer
            self.raw = Tokenizer.from_file(os.path.join(d, "tokenizer.json")); self.kind = "raw"
            self.vocab = self.raw.get_vocab()
        else:
            self.hf = hf; self.kind = "hf"; self.vocab = hf.get_vocab()
        self.byte_ids = {i for t, i in self.vocab.items() if BYTE_TOKEN.match(t)}
        self.fingerprint_src = "\n".join(sorted(self.vocab))

    def encode(self, texts):
        texts = list(texts)
        if self.kind == "tiktoken":
            return [self.enc.encode(t, disallowed_special=()) for t in texts]
        if self.kind == "raw":
            return [e.ids for e in self.raw.encode_batch(texts, add_special_tokens=False)]
        return self.hf(texts, add_special_tokens=False)["input_ids"]


def load(repo):
    return Tok(repo)


def encode(tok, texts):
    return tok.encode(texts)


def main():
    in22 = pd.read_parquet("data/in22_gen_test.parquet")
    langs = list(in22.columns)
    dak = pd.read_csv("data/dakshina.csv").dropna()
    dak = dak[(dak.indic.str.strip() != "") & (dak.roman.str.strip() != "") & (dak.lang_code != "si")]  # Sinhala is not an Indian language

    per_sent, summ, dsum, info = [], [], [], []
    for label, repo, note in TOKENIZERS:
        tok = load(repo)
        vocab = tok.vocab
        fp = hashlib.sha1(tok.fingerprint_src.encode()).hexdigest()[:12]
        info.append({"tokenizer": label, "repo": repo, "loader": tok.kind, "vocab_size": len(vocab), "fingerprint": fp, "note": note})
        unk = tok.unk_id
        byte_ids = tok.byte_ids

        counts = {}
        for lang in langs:
            ids = encode(tok, in22[lang])
            n = np.array([len(x) for x in ids])
            counts[lang] = n
            flat = [i for x in ids for i in x]
            summ.append({
                "tokenizer": label, "lang": lang,
                "tokens": int(n.sum()),
                "words": int(in22[lang].str.split().str.len().sum()),
                "chars": int(in22[lang].str.len().sum()),
                "unk_rate": (sum(i == unk for i in flat) / len(flat)) if unk is not None else 0.0,
                "byte_fallback_rate": (sum(i in byte_ids for i in flat) / len(flat)) if byte_ids else 0.0,
            })
        for lang in langs:
            per_sent.append(pd.DataFrame({"tokenizer": label, "lang": lang,
                                          "sent_id": range(len(in22)), "tokens": counts[lang]}))

        for code, g in dak.groupby("lang_code"):
            nat = sum(len(x) for x in encode(tok, g.indic))
            rom = sum(len(x) for x in encode(tok, g.roman))
            dsum.append({"tokenizer": label, "lang_code": code, "sentences": len(g),
                         "native_tokens": nat, "roman_tokens": rom, "roman_over_native": rom / nat})
        print(f"{label:14s} vocab={len(vocab):>7,}  done")

    s = pd.DataFrame(summ)
    eng = s[s.lang == "eng_Latn"].set_index("tokenizer").tokens
    s["premium_vs_english"] = s.tokens / s.tokenizer.map(eng)
    s["fertility"] = s.tokens / s.words
    s["chars_per_token"] = s.chars / s.tokens
    s.to_csv("results/in22_summary.csv", index=False)
    pd.concat(per_sent).to_parquet("results/in22_per_sentence.parquet", index=False)
    pd.DataFrame(dsum).to_csv("results/dakshina_summary.csv", index=False)
    pd.DataFrame(info).to_csv("results/tokenizer_info.csv", index=False)
    print("saved results/")


if __name__ == "__main__":
    main()
