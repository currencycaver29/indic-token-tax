"""Downloads the datasets and tokenizer files used in the study (about 150 MB)."""
import os, urllib.request
from huggingface_hub import snapshot_download, hf_hub_download
from tokenizers_list import TOKENIZERS

os.makedirs("data", exist_ok=True)
# IN22-Gen (AI4Bharat, CC BY 4.0) - ungated MTEB copy
p = hf_hub_download("mteb/IN22-Gen", "test.parquet", repo_type="dataset")
os.replace(p, "data/in22_gen_test.parquet") if not os.path.exists("data/in22_gen_test.parquet") else None
# Dakshina (Google Research, CC BY-SA 4.0) - CSV copy with native + romanized sentence pairs
p = hf_hub_download("suhani-sarvam/google-dakshina", "google-dakshina.csv", repo_type="dataset")
os.replace(p, "data/dakshina.csv") if not os.path.exists("data/dakshina.csv") else None
# Tokenizer files only (no model weights)
for label, repo, _ in TOKENIZERS:
    snapshot_download(repo, local_dir=f"tok/{repo.replace('/', '__')}",
                      allow_patterns=["tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.txt",
                                      "special_tokens_map.json", "sentencepiece.bpe.model", "spiece.model", "config.json"])
    print("ok", label)
