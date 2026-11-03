"""Tokenizers compared. (label, Hugging Face repo, notes)"""
TOKENIZERS = [
    ("GPT-4o",        "Xenova/gpt-4o",                          "counted with OpenAI's official tiktoken (o200k_base)"),
    ("GPT-4",         "Xenova/gpt-4",                           "counted with OpenAI's official tiktoken (cl100k_base)"),
    ("Llama 3",       "unsloth/Llama-3.2-1B",                   "mirror of meta-llama/Llama-3.2-1B tokenizer"),
    ("Llama 4",       "unsloth/Llama-4-Scout-17B-16E-Instruct", "mirror of Meta's Llama 4 Scout tokenizer"),
    ("Gemma 3",       "unsloth/gemma-3-1b-it",                  "mirror of google/gemma-3-1b-it tokenizer"),
    ("Mistral NeMo",  "unsloth/Mistral-Nemo-Base-2407",         "Tekken tokenizer; mirror of mistralai repo"),
    ("Qwen 3",        "Qwen/Qwen3-0.6B",                        ""),
    ("DeepSeek-V3",   "deepseek-ai/DeepSeek-V3",                ""),
    ("Sarvam-1",      "sarvamai/sarvam-1",                      "Indian, Indic-optimised"),
    ("Krutrim-2",     "krutrim-ai-labs/Krutrim-2-instruct",     "Indian"),
    ("IndicBERT v2",  "ai4bharat/IndicBERTv2-MLM-only",         "Indian, encoder"),
    ("MuRIL",         "google/muril-base-cased",                "Google India, encoder"),
    ("mBERT",         "google-bert/bert-base-multilingual-cased", "encoder baseline"),
    ("XLM-R",         "FacebookAI/xlm-roberta-base",            "encoder baseline"),
    ("BLOOM",         "bigscience/bloom-560m",                  "multilingual LLM, trained with Indic data"),
]
