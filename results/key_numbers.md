## Mean premium per tokenizer (22 languages, 95% bootstrap CI)

| tokenizer                |   mean_premium |   ci_low |   ci_high | best_lang   |   best | worst_lang         |   worst |   langs_masked_unk |   vocab_size | origin   | type    |   premium_sarvam10 |   premium_other12 |
|:-------------------------|---------------:|---------:|----------:|:------------|-------:|:-------------------|--------:|-------------------:|-------------:|:---------|:--------|-------------------:|------------------:|
| MuRIL                    |           1.34 |     1.33 |      1.35 | Bengali     |   1.01 | Bodo               |    2.04 |                  2 |       197258 | Indian   | encoder |               1.19 |              1.49 |
| IndicBERT v2             |           1.43 |     1.42 |      1.45 | Bengali     |   1.03 | Manipuri (Meitei)  |    2.88 |                  0 |       250000 | Indian   | encoder |               1.17 |              1.65 |
| XLM-R                    |           1.49 |     1.47 |      1.5  | Nepali      |   1.14 | Bodo               |    2.04 |                  3 |       250002 | Global   | encoder |               1.4  |              1.58 |
| mBERT                    |           2.18 |     2.16 |      2.2  | Urdu        |   1.61 | Assamese           |    2.69 |                  7 |       119547 | Global   | encoder |               2.17 |              2.18 |
| Gemma 3                  |           2.33 |     2.3  |      2.36 | Bengali     |   1.18 | Manipuri (Meitei)  |    8.38 |                  0 |       262145 | Global   | LLM     |               1.91 |              2.68 |
| BLOOM                    |           2.48 |     2.45 |      2.5  | Bengali     |   1.17 | Santali (Ol Chiki) |   12.34 |                  0 |       250680 | Global   | LLM     |               1.35 |              3.42 |
| Sarvam-1                 |           2.81 |     2.77 |      2.84 | Marathi     |   1.06 | Santali (Ol Chiki) |   10.95 |                  0 |        68096 | Indian   | LLM     |               1.28 |              4.08 |
| GPT-4o                   |           3.1  |     3.07 |      3.13 | Hindi       |   1.55 | Santali (Ol Chiki) |   13.16 |                  0 |       200019 | Global   | LLM     |               2.25 |              3.81 |
| Llama 4                  |           3.45 |     3.41 |      3.49 | Hindi       |   1.62 | Manipuri (Meitei)  |   11.42 |                  0 |       201135 | Global   | LLM     |               3.17 |              3.68 |
| Mistral NeMo / Krutrim-2 |           3.65 |     3.61 |      3.7  | Urdu        |   1.7  | Odia               |   13.06 |                  0 |       131072 | Global   | LLM     |               3.36 |              3.9  |
| DeepSeek-V3              |           3.66 |     3.62 |      3.7  | Bengali     |   1.98 | Manipuri (Meitei)  |    8.3  |                  0 |       128815 | Global   | LLM     |               3.62 |              3.7  |
| Qwen 3                   |           5.65 |     5.57 |      5.72 | Urdu        |   2.95 | Manipuri (Meitei)  |   11    |                  0 |       151669 | Global   | LLM     |               6.22 |              5.17 |
| Llama 3                  |           5.68 |     5.61 |      5.74 | Maithili    |   2.43 | Odia               |   12.52 |                  0 |       128256 | Global   | LLM     |               7.01 |              4.57 |
| GPT-4                    |           6.69 |     6.6  |      6.77 | Urdu        |   4.04 | Odia               |   12.75 |                  0 |       100277 | Global   | LLM     |               7.48 |              6.02 |

## Per-language premium (mean across tokenizers, UNK-masked)

| lang               |    0 |
|:-------------------|-----:|
| Hindi              | 1.98 |
| Nepali             | 2.07 |
| Marathi            | 2.14 |
| Maithili           | 2.15 |
| Konkani            | 2.26 |
| Urdu               | 2.3  |
| Bengali            | 2.33 |
| Dogri              | 2.37 |
| Sanskrit           | 2.45 |
| Sindhi (Deva.)     | 2.58 |
| Bodo               | 2.72 |
| Tamil              | 2.97 |
| Gujarati           | 3    |
| Assamese           | 3.01 |
| Telugu             | 3.11 |
| Punjabi            | 3.13 |
| Kannada            | 3.31 |
| Malayalam          | 3.42 |
| Kashmiri           | 3.55 |
| Odia               | 6.09 |
| Santali (Ol Chiki) | 9.41 |
| Manipuri (Meitei)  | 9.86 |

## Romanized / native token ratio (mean across tokenizers)

| lang_code   |    0 |
|:------------|-----:|
| Punjabi     | 0.79 |
| Telugu      | 0.87 |
| Malayalam   | 0.9  |
| Gujarati    | 0.92 |
| Sindhi      | 0.92 |
| Kannada     | 0.93 |
| Urdu        | 0.99 |
| Bengali     | 1.01 |
| Marathi     | 1.01 |
| Hindi       | 1.04 |
| Tamil       | 1.05 |

## Romanized / native per tokenizer (mean across languages)

| tokenizer                |    0 |
|:-------------------------|-----:|
| MuRIL                    | 1.43 |
| IndicBERT v2             | 1.38 |
| XLM-R                    | 1.17 |
| mBERT                    | 0.89 |
| Gemma 3                  | 1.09 |
| BLOOM                    | 1.36 |
| Sarvam-1                 | 1.44 |
| GPT-4o                   | 0.99 |
| Llama 4                  | 0.8  |
| Mistral NeMo / Krutrim-2 | 0.89 |
| DeepSeek-V3              | 0.66 |
| Qwen 3                   | 0.42 |
| Llama 3                  | 0.44 |
| GPT-4                    | 0.34 |

## UNK rates above 2%

| tokenizer   | lang               |   unk_rate |
|:------------|:-------------------|-----------:|
| MuRIL       | Manipuri (Meitei)  |      0.894 |
| MuRIL       | Santali (Ol Chiki) |      0.924 |
| mBERT       | Manipuri (Meitei)  |      0.896 |
| mBERT       | Odia               |      0.812 |
| mBERT       | Kashmiri           |      0.154 |
| mBERT       | Punjabi            |      0.022 |
| mBERT       | Sindhi (Deva.)     |      0.036 |
| mBERT       | Santali (Ol Chiki) |      0.923 |
| mBERT       | Bengali            |      0.038 |
| XLM-R       | Manipuri (Meitei)  |      0.482 |
| XLM-R       | Kashmiri           |      0.036 |
| XLM-R       | Santali (Ol Chiki) |      0.484 |

## Byte-fallback rates above 5%

| tokenizer   | lang               |   byte_fallback_rate |
|:------------|:-------------------|---------------------:|
| Gemma 3     | Manipuri (Meitei)  |                0.696 |
| Gemma 3     | Kashmiri           |                0.087 |
| Sarvam-1    | Kannada            |                0.094 |
| Sarvam-1    | Urdu               |                0.875 |
| Sarvam-1    | Manipuri (Meitei)  |                0.936 |
| Sarvam-1    | Odia               |                0.148 |
| Sarvam-1    | Kashmiri           |                0.895 |
| Sarvam-1    | Punjabi            |                0.089 |
| Sarvam-1    | Sindhi (Deva.)     |                0.122 |
| Sarvam-1    | Assamese           |                0.395 |
| Sarvam-1    | Dogri              |                0.115 |
| Sarvam-1    | Santali (Ol Chiki) |                0.936 |
| Sarvam-1    | Bengali            |                0.154 |
| Sarvam-1    | Gujarati           |                0.082 |

## Premium on Sarvam-1's 10 languages vs other 12

| tokenizer                | type    |   premium_sarvam10 |   premium_other12 |
|:-------------------------|:--------|-------------------:|------------------:|
| MuRIL                    | encoder |               1.19 |              1.49 |
| IndicBERT v2             | encoder |               1.17 |              1.65 |
| XLM-R                    | encoder |               1.4  |              1.58 |
| mBERT                    | encoder |               2.17 |              2.18 |
| Gemma 3                  | LLM     |               1.91 |              2.68 |
| BLOOM                    | LLM     |               1.35 |              3.42 |
| Sarvam-1                 | LLM     |               1.28 |              4.08 |
| GPT-4o                   | LLM     |               2.25 |              3.81 |
| Llama 4                  | LLM     |               3.17 |              3.68 |
| Mistral NeMo / Krutrim-2 | LLM     |               3.36 |              3.9  |
| DeepSeek-V3              | LLM     |               3.62 |              3.7  |
| Qwen 3                   | LLM     |               6.22 |              5.17 |
| Llama 3                  | LLM     |               7.01 |              4.57 |
| GPT-4                    | LLM     |               7.48 |              6.02 |

## NFC normalization effects beyond ±1%

| tokenizer    | lang           |   change |
|:-------------|:---------------|---------:|
| Sarvam-1     | Bengali        |   -0.174 |
| Sarvam-1     | Punjabi        |   -0.1   |
| Mistral NeMo | Bengali        |   -0.075 |
| MuRIL        | Punjabi        |   -0.044 |
| Sarvam-1     | Sindhi (Deva.) |   -0.044 |
| Sarvam-1     | Odia           |   -0.031 |
| MuRIL        | Odia           |   -0.025 |
| MuRIL        | Bengali        |   -0.022 |
| BLOOM        | Punjabi        |   -0.018 |
| Sarvam-1     | Assamese       |   -0.017 |
| MuRIL        | Assamese       |   -0.013 |
| Mistral NeMo | Punjabi        |   -0.013 |
| Sarvam-1     | Dogri          |   -0.012 |
| Sarvam-1     | Sanskrit       |   -0.011 |
| Mistral NeMo | Assamese       |   -0.01  |
| BLOOM        | Sindhi (Deva.) |   -0.01  |
| mBERT        | Dogri          |    0.01  |
| IndicBERT v2 | Bengali        |    0.011 |
| GPT-4        | Sindhi (Deva.) |    0.012 |
| GPT-4o       | Bengali        |    0.013 |
| GPT-4        | Punjabi        |    0.013 |
| Llama 3      | Punjabi        |    0.013 |
| mBERT        | Sanskrit       |    0.019 |
| mBERT        | Assamese       |    0.022 |
| GPT-4        | Bengali        |    0.023 |
| Llama 3      | Bengali        |    0.023 |
| mBERT        | Sindhi (Deva.) |    0.056 |
| mBERT        | Punjabi        |    0.073 |
| mBERT        | Bengali        |    0.076 |

## Duplicates removed

Krutrim-2 has the same vocabulary as Mistral NeMo
