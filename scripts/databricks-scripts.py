# Copy of script in Databricks notebook, purpose of this is for AI to analyze and understand the script



%pip install pinecone sentence-transformers

import time
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# 1. Get the uploaded filename from the Job Trigger parameter
try:
    blob_name = dbutils.widgets.get("blob_name")
except:
    blob_name = "finance-data.csv"  # Default fallback for manual testing

print(f"Processing file: {blob_name}")


# 2. Polling loop: Wait for Fivetran to sync the file to perfectly separate data
print("Waiting for Fivetran sync...")
max_attempts = 120 # 10 mins max
for attempt in range(max_attempts):
    count_df = spark.sql(f"SELECT COUNT(*) as count FROM azure_blob_storage.smfinancefivetran WHERE _file LIKE '%{blob_name}'").toPandas()
    if count_df["count"][0] > 0:
        print(f"✅ Found {count_df['count'][0]} rows for {blob_name}. Proceeding...")
        break
    if attempt == max_attempts - 1:
        raise TimeoutError(f"File {blob_name} did not appear in Fivetran sync after 10 minutes.")
    time.sleep(5)


# 3. Fetch ONLY the newly uploaded file to avoid reprocessing the whole database
df = spark.sql(f"SELECT * FROM azure_blob_storage.smfinancefivetran WHERE _file LIKE '%{blob_name}'").toPandas()

# Drop Fivetran metadata columns except _file (we use _file for Pinecone's source metadata)
df = df.drop(columns=["_fivetran_synced", "_modified", "_line"], errors="ignore")

print(f"✅ Loaded {len(df)} rows to embed")


# 4. Convert rows into concatenated text chunks
def combine_row_to_text(row):
    return " ".join([f"{col}: {row[col]}" for col in df.columns if col != "_file"])

df["combined_text"] = df.apply(combine_row_to_text, axis=1)

chunks_data = []
for i, row in df.iterrows():
    chunks_data.append({
        "id": f"{blob_name}_row_{i}", # Unique ID prefix per file to prevent overwriting
        "text": row["combined_text"],
        "source": row.get("_file", blob_name) # Ensure source metadata matches perfectly
    })

chunks_df = pd.DataFrame(chunks_data)
print(f"✅ Created {len(chunks_df)} chunks")


# 5. Initialize Pinecone and Vectorize
try:
    PINECONE_API_KEY = dbutils.secrets.get(scope="pinecone_scope", key="api_key")
except:
    # Fallback to hardcoded string if secrets scope isn't configured for testing
    PINECONE_API_KEY = "pcsk_DgvsU_QSsoniNnraSq9aqq4PthPu7DKhFZLudUy4mpGX9rTaT8AGpekt2dH9QKWrurT7n"

INDEX_NAME = "financial-docs"
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks_df["text"].tolist())


# 6. Upload vectors to Pinecone in batches
vectors = []
for i, row in chunks_df.iterrows():
    vectors.append({
        "id": row["id"],
        "values": embeddings[i].tolist(),
        "metadata": {
            "text": row["text"],
            "source": row["source"]
        }
    })

batch_size = 100
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i+batch_size]
    index.upsert(vectors=batch)
    print(f"✅ Uploaded batch {i//batch_size + 1}")

print(f"✅ SUCCESS: Total {len(vectors)} vectors for {blob_name} uploaded to Pinecone!")
