import os
import re
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
DATA_FILE = "data/arxiv_subset.parquet"

POOL_K = 20   # скільки кандидатів беремо з кожного методу, перш ніж застосувати RRF
TOP_K = 5     # скільки результатів показуємо в підсумку
RRF_K = 60    # константа згладжування в класичній формулі RRF

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet(DATA_FILE).reset_index(drop=True)
print(f"Завантажено {len(df)} статей")


# ---------------------------------------------------------------------------
# 1. Локальний BM25-індекс за заголовками й анотаціями всіх статей
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Проста токенізація: нижній регістр + послідовності буквено-цифрових символів."""
    return re.findall(r"[a-z0-9]+", text.lower())


tokenized_corpus = [tokenize(f"{row.title} {row.abstract}") for row in df.itertuples()]
bm25 = BM25Okapi(tokenized_corpus)


# ---------------------------------------------------------------------------
# 2. Кодування запиту для векторного пошуку в Pinecone (як у 04_search.py)
# ---------------------------------------------------------------------------
def encode_query(text: str) -> list[float]:
    return model.encode(text, normalize_embeddings=True).tolist()


def row_to_doc(row: pd.Series) -> dict:
    return {
        "arxiv_id": str(row["id"]),
        "title": row["title"],
        "authors": row["authors"],
        "year": int(row["year"]),
        "category": row["category"],
    }


# ---------------------------------------------------------------------------
# 3. Функції пошуку: BM25, векторний (Pinecone), гібридний (RRF)
# ---------------------------------------------------------------------------
def search_bm25(query: str, top_k: int = POOL_K) -> list[dict]:
    """Лексичний пошук за BM25. Повертає ранжований список документів
    з полем bm25_score (документи з нульовим збігом відкидаються)."""
    scores = bm25.get_scores(tokenize(query))
    order = scores.argsort()[::-1][:top_k]
    results = []
    for rank, idx in enumerate(order, start=1):
        if scores[idx] <= 0:
            continue
        doc = row_to_doc(df.iloc[idx])
        doc["rank"] = rank
        doc["bm25_score"] = float(scores[idx])
        results.append(doc)
    return results


def search_vector(query: str, top_k: int = POOL_K) -> list[dict]:
    """Семантичний пошук у Pinecone на SPECTER2-ембеддингах. Повертає
    ранжований список документів з полем vector_score."""
    query_vector = encode_query(query)
    response = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    results = []
    for rank, match in enumerate(response["matches"], start=1):
        meta = match["metadata"]
        results.append({
            "arxiv_id": meta["arxiv_id"],
            "title": meta["title"],
            "authors": meta["authors"],
            "year": int(meta["year"]),
            "category": meta["category"],
            "rank": rank,
            "vector_score": float(match["score"]),
        })
    return results


def reciprocal_rank_fusion(
    bm25_results: list[dict],
    vector_results: list[dict],
    k: int = RRF_K,
    top_k: int = TOP_K,
) -> list[dict]:
    """Об'єднує два ранжовані списки за формулою Reciprocal Rank Fusion:
        RRF(d) = sum_за_методами( 1 / (k + rank_методу(d)) )
    Документ, знайдений в обох списках, отримує суму двох доданків і тому
    зазвичай підіймається вище за документ з лише одного методу — так
    RRF узгоджує різні шкали й розподіли скорів BM25 і косинусної схожості,
    працюючи виключно з рангами."""
    docs_by_id: dict[str, dict] = {}
    rrf_scores: defaultdict = defaultdict(float)

    for doc in bm25_results:
        arxiv_id = doc["arxiv_id"]
        docs_by_id.setdefault(arxiv_id, {**doc})
        docs_by_id[arxiv_id]["bm25_score"] = doc["bm25_score"]
        rrf_scores[arxiv_id] += 1.0 / (k + doc["rank"])

    for doc in vector_results:
        arxiv_id = doc["arxiv_id"]
        docs_by_id.setdefault(arxiv_id, {**doc})
        docs_by_id[arxiv_id]["vector_score"] = doc["vector_score"]
        rrf_scores[arxiv_id] += 1.0 / (k + doc["rank"])

    ranked_ids = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:top_k]

    fused = []
    for rank, arxiv_id in enumerate(ranked_ids, start=1):
        doc = dict(docs_by_id[arxiv_id])
        doc["rank"] = rank
        doc["rrf_score"] = rrf_scores[arxiv_id]
        fused.append(doc)
    return fused


# ---------------------------------------------------------------------------
# 4. Виведення й порівняння результатів
# ---------------------------------------------------------------------------
def print_results(results: list[dict], heading: str, score_key: str) -> None:
    print(f"\n=== {heading} ===")
    if not results:
        print("(нічого не знайдено)")
        return
    for doc in results[:TOP_K]:
        print(f"{doc['rank']}. [{score_key}={doc[score_key]:.4f}] {doc['title']}")
        print(f"   автори: {doc['authors'][:80]} | категорія: {doc['category']} | рік: {doc['year']}")


def print_fused_results(results: list[dict], heading: str) -> None:
    print(f"\n=== {heading} ===")
    if not results:
        print("(нічого не знайдено)")
        return
    for doc in results:
        bm25_part = f"bm25={doc['bm25_score']:.2f}" if "bm25_score" in doc else "bm25=—"
        vector_part = f"vector={doc['vector_score']:.4f}" if "vector_score" in doc else "vector=—"
        print(f"{doc['rank']}. [RRF={doc['rrf_score']:.4f}] ({bm25_part}, {vector_part}) {doc['title']}")
        print(f"   автори: {doc['authors'][:80]} | категорія: {doc['category']} | рік: {doc['year']}")


def run_query(query: str) -> None:
    print(f"\n{'#' * 80}\nЗапит: \"{query}\"\n{'#' * 80}")

    bm25_results = search_bm25(query)
    vector_results = search_vector(query)
    fused_results = reciprocal_rank_fusion(bm25_results, vector_results)

    print_results(bm25_results, "Топ-5 BM25", "bm25_score")
    print_results(vector_results, "Топ-5 векторний пошук (Pinecone / SPECTER2)", "vector_score")
    print_fused_results(fused_results, "Топ-5 гібридний пошук (RRF)")


# ---------------------------------------------------------------------------
# 5. Демонстраційні запити
# ---------------------------------------------------------------------------
demo_queries = [
    "BERT fine-tuning",                                   # точний термін
    "Yann LeCun convolutional networks",                  # ім'я автора
    "making computers understand human emotions from text",  # перефразування без явних термінів
]

for demo_query in demo_queries:
    run_query(demo_query)
