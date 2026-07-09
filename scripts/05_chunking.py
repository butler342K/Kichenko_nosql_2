import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

DATA_FILE = "data/arxiv_subset.parquet"
TOP_N_ARTICLES = 30

INDEX_NAME_FIXED = "arxiv-chunks-fixed"
INDEX_NAME_SEMANTIC = "arxiv-chunks-semantic"
CLOUD = "aws"
REGION = "us-east-1"
BATCH_SIZE = 100

FIXED_CHUNK_SIZE = 60   # слів у чанку
FIXED_OVERLAP = 15      # слів перекриття між сусідніми чанками

SEMANTIC_MAX_WORDS = 60  # максимум слів у семантичному чанку

TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)


# ---------------------------------------------------------------------------
# 1. Вибір 30 статей із найдовшими анотаціями
# ---------------------------------------------------------------------------
df = pd.read_parquet(DATA_FILE)
df["abstract_words"] = df["abstract"].str.split().apply(len)
top_df = df.nlargest(TOP_N_ARTICLES, "abstract_words").reset_index(drop=True)
print(f"Відібрано {len(top_df)} статей із найдовшими анотаціями")
print(f"Довжина анотацій (слів): min={top_df['abstract_words'].min()}, "
      f"max={top_df['abstract_words'].max()}, mean={top_df['abstract_words'].mean():.1f}")


# ---------------------------------------------------------------------------
# 2. Дві стратегії чанкінгу
# ---------------------------------------------------------------------------
def chunk_fixed(text: str, chunk_size: int = FIXED_CHUNK_SIZE, overlap: int = FIXED_OVERLAP) -> list[str]:
    """Розбиває текст на чанки фіксованої довжини (у словах) з перекриттям."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def split_sentences(text: str) -> list[str]:
    """Наївний розбивач на речення за розділовими знаками .!? з пробілом після."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_semantic(text: str, max_words: int = SEMANTIC_MAX_WORDS) -> list[str]:
    """Об'єднує послідовні речення в чанк, поки не буде перевищено ліміт слів,
    щоб не розривати речення посередині та зберегти зміст."""
    sentences = split_sentences(text)
    chunks = []
    current_sentences: list[str] = []
    current_words = 0
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current_sentences and current_words + sentence_words > max_words:
            chunks.append(" ".join(current_sentences))
            current_sentences = []
            current_words = 0
        current_sentences.append(sentence)
        current_words += sentence_words
    if current_sentences:
        chunks.append(" ".join(current_sentences))
    return chunks


def build_chunk_records(articles: pd.DataFrame, chunk_fn) -> list[dict]:
    """Формує список записів {id, text, metadata} для всіх статей за заданою
    стратегією чанкінгу."""
    records = []
    for _, row in articles.iterrows():
        chunks = chunk_fn(row["abstract"])
        for chunk_num, chunk_text in enumerate(chunks):
            records.append({
                "id": f"{row['id']}_{chunk_num}",
                "text": chunk_text,
                "metadata": {
                    "arxiv_id": str(row["id"]),
                    "title": row["title"],
                    "text": chunk_text,
                    "chunk_num": chunk_num,
                    "year": int(row["year"]),
                    "category": row["category"],
                },
            })
    return records


fixed_records = build_chunk_records(top_df, chunk_fixed)
semantic_records = build_chunk_records(top_df, chunk_semantic)
print(f"Fixed-size чанків: {len(fixed_records)}")
print(f"Semantic чанків: {len(semantic_records)}")


# ---------------------------------------------------------------------------
# 3. Створення окремих індексів у Pinecone
# ---------------------------------------------------------------------------
def get_or_create_index(name: str):
    if name not in pc.list_indexes().names():
        pc.create_index(
            name=name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        print(f"Створено індекс: {name}")
    return pc.Index(name)


index_fixed = get_or_create_index(INDEX_NAME_FIXED)
index_semantic = get_or_create_index(INDEX_NAME_SEMANTIC)


# ---------------------------------------------------------------------------
# 4-5. Ембеддинги для кожного чанка й завантаження в Pinecone батчами
# ---------------------------------------------------------------------------
def embed_and_upload(records: list[dict], index, index_name: str) -> None:
    texts = [rec["text"] for rec in records]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    for start in tqdm(range(0, len(records), BATCH_SIZE), desc=f"Завантаження в {index_name}"):
        batch_records = records[start:start + BATCH_SIZE]
        batch_vectors = embeddings[start:start + BATCH_SIZE]
        vectors = [
            {
                "id": rec["id"],
                "values": vector.tolist(),
                "metadata": rec["metadata"],
            }
            for rec, vector in zip(batch_records, batch_vectors)
        ]
        index.upsert(vectors=vectors)

    print(f"Завантажено {len(records)} чанків в індекс '{index_name}'")


embed_and_upload(fixed_records, index_fixed, INDEX_NAME_FIXED)
embed_and_upload(semantic_records, index_semantic, INDEX_NAME_SEMANTIC)


# ---------------------------------------------------------------------------
# 6. Пошук по чанках
# ---------------------------------------------------------------------------
def search_chunks(index, query: str, top_k: int = TOP_K):
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return result["matches"]


def print_chunk_matches(matches, heading: str) -> None:
    print(f"\n=== {heading} ===")
    if not matches:
        print("(нічого не знайдено)")
        return
    for rank, match in enumerate(matches, start=1):
        meta = match["metadata"]
        snippet = meta["text"][:200].rstrip()
        print(f"{rank}. [score={match['score']:.4f}] {meta['title']} (чанк #{meta['chunk_num']})")
        print(f"   {snippet}...")


test_queries = [
    "graph neural networks for molecular property prediction",
    "reinforcement learning for robotic control",
    "transformer models for natural language understanding",
]

for query in test_queries:
    fixed_matches = search_chunks(index_fixed, query)
    semantic_matches = search_chunks(index_semantic, query)
    print_chunk_matches(fixed_matches, f"Fixed-size чанки: '{query}'")
    print_chunk_matches(semantic_matches, f"Semantic чанки: '{query}'")
