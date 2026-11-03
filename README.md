# The Token Tax: Indian languages across 14 tokenizers

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23885241.svg)](https://doi.org/10.5281/zenodo.23885241)

**Paper:** *The Token Tax on Indian Languages: Measuring Tokenizer Premiums Across All 22 Scheduled Languages and 14 Tokenizers*. Divij Karanam, 2026. Preprint on Zenodo: https://doi.org/10.5281/zenodo.23885241 ([PDF in this repo](paper/main.pdf))

<details><summary>Cite</summary>

```bibtex
@misc{karanam2026tokentax,
  author    = {Karanam, Divij},
  title     = {The Token Tax on Indian Languages: Measuring Tokenizer Premiums Across All 22 Scheduled Languages and 14 Tokenizers},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23885241},
  url       = {https://doi.org/10.5281/zenodo.23885241}
}
```
</details>

How many more tokens does the same sentence need in each of the 22 scheduled Indian languages than in English?
Plus: does typing in English letters (romanization) help, and does Unicode normalization matter?

## Run it

```
pip install -r requirements.txt
python download.py        # datasets + tokenizer files (~150 MB)
python measure.py         # token counts -> results/  (~4 min on 4 CPUs)
python normalization.py   # Unicode NFC experiment -> results/nfc_effect.csv
python analyze.py         # charts + tables -> results/
```

## Data

- **IN22-Gen** (AI4Bharat, from the IndicTrans2 paper; CC BY 4.0): 1,024 sentences professionally translated into 22 Indian languages + English. Loaded from the ungated `mteb/IN22-Gen` copy.
- **Dakshina** (Google Research, Roark et al. 2020; CC BY-SA 4.0): native-script sentences with human romanizations, 11 Indian languages used (Sinhala excluded). Loaded from the `suhani-sarvam/google-dakshina` CSV copy; empty rows dropped.

## Tokenizers and how each was loaded

See `tokenizers_list.py` and `results/tokenizer_info.csv` (loader column).

- **GPT-4o / GPT-4:** OpenAI's official `tiktoken` (`o200k_base`, `cl100k_base`).
- **All others with a `tokenizer.json`:** the raw Hugging Face `tokenizers` file. Cross-checked: identical counts to `transformers.AutoTokenizer` for every one of these.
- **MuRIL** (WordPiece `vocab.txt` only): `transformers` BertTokenizerFast.
- Llama, Gemma and Mistral tokenizers come from unsloth's ungated mirrors of the official files.

**Important verification note.** With the `transformers` version used here, loading the community GPT tokenizer conversions (`Xenova/gpt-4o`, `Xenova/gpt-4`) through `AutoTokenizer` produced wrong counts (GPT-4o +46%, GPT-4 +4% vs official tiktoken). The raw `tokenizer.json` matched tiktoken exactly on 3,450/3,450 test sentences. That's why tiktoken is used for the GPT models. Also, the `fix_mistral_regex=True` flag suggested by a `transformers` warning changed nothing for Mistral NeMo / Krutrim-2 and wrongly changed counts for BERT-style tokenizers, so it is not used.

## Metrics

- **Token premium** = tokens for the 1,024 sentences in language X ÷ tokens for the same sentences in English, same tokenizer.
- **Fertility** = tokens ÷ whitespace-separated words.
- **UNK masking:** a tokenizer × language cell is excluded when more than 2% of its tokens are `[UNK]` (the text is being discarded, so the count is meaningless). Affected: mBERT 7 languages, XLM-R 3, MuRIL 2.
- **Byte fallback rate:** share of tokens that are raw-byte tokens (`<0xNN>`), which shows the script isn't in the vocabulary.
- **95% CIs:** paired bootstrap over sentences (1,000 resamples).
- **Duplicates:** tokenizers with identical vocabularies are reported once (Krutrim-2 = Mistral NeMo).
- **Romanized ÷ native:** Dakshina token counts in human romanization ÷ native script, same sentences.

## Outputs (results/)

| File | Contents |
|---|---|
| `in22_summary.csv` | tokens, words, chars, premium, fertility, UNK and byte-fallback rates per tokenizer × language |
| `in22_per_sentence.parquet` | per-sentence token counts |
| `mean_premium.csv` | average premium per tokenizer with CI, best/worst language, Sarvam-10 vs other-12 split |
| `premium_matrix.csv` | tokenizer × language premium (unmasked) |
| `romanized_ratio_matrix.csv` | romanized ÷ native ratios |
| `nfc_effect.csv` | token change after Unicode NFC normalization |
| `key_numbers.md` | all headline numbers |
| `fig1`–`fig5` PNGs | charts |

## Limitations

Token counts measure cost and context use, not model quality. IN22-Gen is formal text; chat or code-mixed text may differ. Premiums are relative to each tokenizer's own English count.

## Environment

4 vCPU Intel Xeon @ 2.80 GHz, Linux, Python 3.13, `tokenizers` + `transformers` + `tiktoken` (versions in `results/versions.txt`).
