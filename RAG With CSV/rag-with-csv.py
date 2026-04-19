import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# df = pd.read_csv('option-chain-ED-NIFTY-21-Apr-2026.csv')
df = pd.read_csv('sensex.csv')
# print(df.head())
# convert rows into text
documents = df.astype(str).apply(lambda row: " | ".join(row), axis = 1).tolist()
# print(documents[60:70])

model = SentenceTransformer('all-MiniLM-L6-V2')
embeddings = model.encode(documents)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

query = "What was the highest closing price of Sensex in the dataset?"
query_vector = model.encode([query])
D, I = index.search(query_vector, k=3)
results = [documents[i] for i in I[0]]
print(results)
