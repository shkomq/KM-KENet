"""Download the full Kurdish Medical Corpus (KMC) V1 repository from Hugging Face.

The pipeline only needs the consolidated JSON file; for that, prefer
    cd KM-KENet && python -m setup.download_assets --only dataset
This script mirrors the whole dataset repository (including the ~1.3 GB of source PDFs).

    python download_kmc.py                 # public access
    HF_TOKEN=hf_xxx python download_kmc.py # if you need authenticated access
"""

import os

from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="shkomq/Kurdish_Medical_Corpus_KMC",
    repo_type="dataset",
    token=os.environ.get("HF_TOKEN") or None,
    local_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Kurdish_Medical_Corpus_KMC"),
)
print("Downloaded to:", path)
