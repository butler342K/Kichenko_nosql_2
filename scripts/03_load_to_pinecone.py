
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
CLOUD = "aws"
REGION = "us-east-1"
BATCH_SIZE = 200   # Pinecone рекомендує батчі до 200 векторів

df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)
assert len(df) == len(embeddings), "Кількість записів і ембеддингів не збігається"
print(f"Завантажено {len(df)} записів, розмірність ембеддингів: {embeddings.shape[1]}")

# Ініціалізація клієнта
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Створюємо індекс (якщо не існує)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=embeddings.shape[1],
        metric="cosine",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )
    print(f"Створено індекс: {INDEX_NAME}")

index = pc.Index(INDEX_NAME)

for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Завантаження в Pinecone"):
    batch_df = df.iloc[start:start + BATCH_SIZE]
    batch_vectors = embeddings[start:start + BATCH_SIZE]

    vectors = [
        {
            "id": f"paper_{start + i}",
            "values": vector.tolist(),
            "metadata": {
                "arxiv_id": str(row["id"]),
                "title": row["title"],
                "abstract": row["abstract"][:500],
                "authors": row["authors"][:200],
                "year": int(row["year"]),
                "category": row["category"],
            },
        }
        for i, ((_, row), vector) in enumerate(zip(batch_df.iterrows(), batch_vectors))
    ]
    index.upsert(vectors=vectors)

print(f"Завантажено {len(df)} векторів в індекс '{INDEX_NAME}'")
