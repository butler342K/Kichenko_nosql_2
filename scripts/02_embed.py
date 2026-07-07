import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_FILE  = "../data/arxiv_subset.parquet"
OUTPUT_DIR  = "../embeddings"
OUTPUT_FILE = f"{OUTPUT_DIR}/embeddings.npy"

df = pd.read_parquet(INPUT_FILE)
print(f"Завантажено записів: {len(df)}")

texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()

# Модель завантажується з Hugging Face автоматично при першому виклику
model = SentenceTransformer("allenai/specter2_base")

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
)

print(f"Оброблено текстів: {len(embeddings)}")
print(f"Розмірність ембеддингів: {embeddings.shape[1]}")
print(f"Норма першого ембеддингу: {np.linalg.norm(embeddings[0]):.4f}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(OUTPUT_FILE, embeddings)
print(f"Збережено в {OUTPUT_FILE}")
