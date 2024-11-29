from sentence_transformers import SentenceTransformer

# Load a powerful general-purpose embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, good quality

# Get embeddings
embeddings = model.encode("Gas")
print(embeddings.shape)