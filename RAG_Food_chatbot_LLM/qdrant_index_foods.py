from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# === Setup ===
BASE_DIR = Path(__file__).parent
csv_path = BASE_DIR / "food_embeddings.csv"

# ✅ Load food data
df = pd.read_csv(csv_path)
if not {"name", "description"}.issubset(df.columns):
    raise ValueError("CSV must have 'name' and 'description' columns.")

# ✅ Load embedder
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = df["description"].astype(str).tolist()
print(f"🔢 Embedding {len(texts)} food descriptions…")
embeddings = model.encode(texts, show_progress_bar=True)

# ✅ Connect to Qdrant (local file-based DB)
QDRANT_PATH = str(BASE_DIR / "qdrant_data")
client = QdrantClient(path=QDRANT_PATH)

COLLECTION_NAME = "food-embeddings"
DIM = embeddings.shape[1]

# Clean old collection if exists
if COLLECTION_NAME in client.get_collections().collections:
    client.delete_collection(collection_name=COLLECTION_NAME)
    print("🧹 Old Qdrant collection deleted.")

# Create new collection
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=DIM, distance=Distance.COSINE)
)

# ✅ Prepare & upload in batches
print("📦 Uploading to Qdrant...")
points = []
for idx, (name, desc, emb) in enumerate(zip(df["name"], df["description"], embeddings)):
    points.append(
        PointStruct(
            id=idx,
            vector=emb.tolist(),
            payload={"name": name, "description": desc}
        )
    )

client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"✅ Successfully indexed {len(points)} food items in Qdrant.")
