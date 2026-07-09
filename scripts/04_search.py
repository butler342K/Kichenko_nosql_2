import os
from datetime import date

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5
DATA_FILE = "data/arxiv_subset.parquet"
EMBEDDINGS_FILE = "embeddings/embeddings.npy"

# ---------------------------------------------------------------------------
# 1. Підключення до Pinecone і завантаження моделі
# ---------------------------------------------------------------------------
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)


# ---------------------------------------------------------------------------
# 2. Кодування запиту в ембеддинг
# ---------------------------------------------------------------------------
def encode_query(text: str) -> list[float]:
    """Кодує текстовий запит у вектор тієї ж розмірності й нормалізації,
    що й ембеддинги статей у 02_embed.py (normalize_embeddings=True)."""
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def print_matches(matches, heading: str) -> None:
    print(f"\n=== {heading} ===")
    if not matches:
        print("(нічого не знайдено за цим фільтром)")
        return
    for rank, match in enumerate(matches, start=1):
        meta = match["metadata"]
        snippet = meta["abstract"][:220].rstrip() + "..."
        print(f"{rank}. [score={match['score']:.4f}] {meta['title']}")
        print(f"   категорія: {meta['category']} | рік: {int(meta['year'])}")
        print(f"   {snippet}")


# ---------------------------------------------------------------------------
# 3. Чистий семантичний пошук
# ---------------------------------------------------------------------------
query = "teaching machines to recognize objects in pictures"
query_vector = encode_query(query)

semantic_result = index.query(vector=query_vector, top_k=TOP_K, include_metadata=True)
print_matches(semantic_result["matches"], f"Семантичний пошук: '{query}'")


# ---------------------------------------------------------------------------
# 4. Пошук з фільтрацією за метаданими
# ---------------------------------------------------------------------------
rl_query = "reinforcement learning for sequential decision making"
rl_vector = encode_query(rl_query)
current_year = date.today().year
year_from = current_year - 5

# Приклад A: reinforcement learning за останні 5 років, категорія cs.LG
filter_a = {
    "category": {"$eq": "cs.LG"},
    "year": {"$gte": year_from},
}
result_a = index.query(vector=rl_vector, top_k=TOP_K, include_metadata=True, filter=filter_a)
print_matches(
    result_a["matches"],
    f"Приклад A: '{rl_query}' | категорія=cs.LG, рік>={year_from}",
)

# Приклад B: старіші статті (до 2015), будь-яка категорія
filter_b = {"year": {"$lt": 2015}}
result_b = index.query(vector=rl_vector, top_k=TOP_K, include_metadata=True, filter=filter_b)
print_matches(
    result_b["matches"],
    f"Приклад B: '{rl_query}' | рік<2015, будь-яка категорія",
)

print(
    """
Порівняння прикладів A і B:
- Приклад A звужує пошук до однієї предметної області (cs.LG) і лише свіжих
  статей — видача складається зі статей саме про RL у сенсі machine learning
  (агенти, policy, reward), бо категорія відсікає все, що не належить cs.LG.
- Приклад B обмежує лише рік (до 2015) і не чіпає категорію, тому у видачі
  можуть з'явитися старіші статті з фізики, теорії керування чи математики,
  де слова "reinforcement"/"learning" вживаються в іншому, не-ML сенсі, або
  ранні (до буму глибокого навчання) роботи з класичного RL. Тобто той самий
  запит дає змістовно іншу видачу залежно від того, яке поле метаданих
  використовується для звуження пошуку.
"""
)


# ---------------------------------------------------------------------------
# 5. Порівняння метрик схожості на локальних ембеддингах
# ---------------------------------------------------------------------------
df = pd.read_parquet(DATA_FILE)
embeddings = np.load(EMBEDDINGS_FILE)
assert len(df) == len(embeddings), "Кількість записів і ембеддингів не збігається"

query_vec = np.array(query_vector)  # той самий запит, що й у п.3

norms = np.linalg.norm(embeddings, axis=1)
cosine_sim = (embeddings @ query_vec) / (norms * np.linalg.norm(query_vec))
dot_product = embeddings @ query_vec
l2_distance = np.linalg.norm(embeddings - query_vec, axis=1)


def print_top_k(scores: np.ndarray, ascending: bool, label: str) -> list[str]:
    order = np.argsort(scores)
    if not ascending:
        order = order[::-1]
    order = order[:TOP_K]
    print(f"\n--- Топ-{TOP_K} за {label} ---")
    for rank, idx in enumerate(order, start=1):
        row = df.iloc[idx]
        print(f"{rank}. [{scores[idx]:.4f}] {row['title']} ({row['category']}, {row['year']})")
    return df.iloc[order]["id"].tolist()


ids_cosine = print_top_k(cosine_sim, ascending=False, label="cosine similarity")
ids_dot = print_top_k(dot_product, ascending=False, label="dot product")
ids_l2 = print_top_k(l2_distance, ascending=True, label="L2-distance (менше = ближче)")

print(
    f"""
Порівняння метрик:
- Ембеддинги SPECTER2 у цьому проєкті нормалізовані до одиничної довжини
  (normalize_embeddings=True у 02_embed.py). Для нормованих векторів
  cosine similarity і dot product збігаються математично:
      cos(q, x) = (q·x) / (|q||x|) = q·x,  якщо |q| = |x| = 1.
  Топ-5 за cosine і за dot product тут ідентичні: {ids_cosine == ids_dot}.
- L2-відстань для нормованих векторів пов'язана з cosine формулою
      ||q - x||^2 = 2 - 2 * cos(q, x),
  тобто є монотонно спадною функцією від cosine similarity. Тому топ-5 за
  L2 (за зростанням відстані) збігається з топ-5 за cosine (за спаданням
  схожості): {ids_cosine == ids_l2}.
- Висновок: коли вектори нормалізовані, вибір метрики (cosine/dot/L2) не
  впливає на ранжування — усі три впорядковують статті однаково. Різниця
  проявилася б лише для ненормованих ембеддингів, де довжина вектора
  (пов'язана, наприклад, з довжиною чи "впевненістю" тексту) впливала б на
  dot product і на L2, але не на cosine.
"""
)
