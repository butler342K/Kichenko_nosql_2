from sentence_transformers import SentenceTransformer
import numpy as np
from itertools import combinations

# Модель завантажується з Hugging Face автоматично при першому виклику
model = SentenceTransformer("allenai/specter2_base")

sentences = [
    "The cat sat on the mat.",
    "A feline rested on a rug.",
    "The stock market crashed today.",
]

embeddings = model.encode(sentences)

print(f"Розмірність вектора:{embeddings.shape[1]}")
print(f"Тип даних:{embeddings.dtype}")
print(f"Перші 5 компонент першого вектора:{embeddings[0][:5]}")

# --- Евклідова відстань (L2) ---
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

print("\\nЕвклідові відстані між реченнями:")
for i, j in combinations(range(len(sentences)), 2):
    dist = euclidean_distance(embeddings[i], embeddings[j])
    print(f"{i} ↔{j}:{dist:.4f}")
