import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os

# Initialize ChromaDB with telemetry off
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

# Use OpenAI embeddings
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

# Collection name
collection = chroma_client.get_or_create_collection(
    name="resume_templates",
    embedding_function=openai_ef
)

def add_template_to_vectorstore(title: str, content: str, metadata: dict):
    """Add a single resume template to ChromaDB"""
    try:
        collection.add(
            documents=[content],
            ids=[title],
            metadatas=[metadata]
        )
    except Exception as e:
        print(f"Error adding template {title} to ChromaDB: {e}")

def search_similar_template(text: str, top_k=3):
    """Semantic search for similar resume templates"""
    try:
        results = collection.query(
            query_texts=[text],
            n_results=top_k
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        matches = []
        for i, doc in enumerate(documents):
            file_id = metadatas[i].get("file_id")
            matches.append({
                "template_number": i + 1,
                "template_preview_text": doc[:500],
                "template_file_id": file_id,
                "download_url": f"/download_template_by_id/{file_id}" if file_id else None
            })

        return matches

    except Exception as e:
        print(f"Search error: {e}")
        return []