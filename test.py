import chromadb

# Connect to the same persistent DB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("legal_knowledge")

# Retrieve all metadata (only ids, not full text)
data = collection.get()
print(f" Total sections loaded: {len(data['ids'])}")


col = client.get_collection("legal_knowledge")
print(len(col.get()["documents"]))
print("Sample metadata:", col.get(limit=1)["metadatas"][0])
