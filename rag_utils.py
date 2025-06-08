import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ChromaDB with persistent storage
chroma_client = chromadb.PersistentClient(path="./chroma_db")

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
        logger.info(f"Successfully added template {title} to ChromaDB")
    except Exception as e:
        logger.error(f"Error adding template {title} to ChromaDB: {e}")
        raise

def search_similar_template(text: str, top_k=3, score_threshold=0.5):
    """
    Semantic search for similar resume templates with enhanced metadata and scoring
    
    Args:
        text (str): The text to search against (usually job description)
        top_k (int): Number of results to return
        score_threshold (float): Minimum similarity score (0-1)
    
    Returns:
        list: List of template matches with metadata and scores
    """
    try:
        logger.info(f"Searching for templates with text length: {len(text)}")
        logger.info(f"Using score threshold: {score_threshold}")
        
        # Query with distances to get similarity scores
        results = collection.query(
            query_texts=[text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        logger.info(f"Found {len(documents)} initial matches")
        
        matches = []
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            # Convert distance to similarity score (ChromaDB uses L2 distance)
            # Lower distance = higher similarity
            similarity_score = 1.0 - min(distance, 1.0)
            
            logger.info(f"Template {i+1} - Distance: {distance:.3f}, Similarity: {similarity_score:.3f}")
            
            if similarity_score < score_threshold:
                logger.info(f"Skipping template {i+1} due to low similarity score: {similarity_score:.3f}")
                continue

            file_id = metadata.get("file_id")
            title = metadata.get("title", f"Template {i+1}")
            filename = metadata.get("filename", "unknown.pdf")
            
            logger.info(f"Processing match: {title} (ID: {file_id}, Score: {similarity_score:.3f})")
            
            match = {
                "template_number": i + 1,
                "template_title": title,
                "template_filename": filename,
                "template_preview_text": doc[:500] + "...",  # Truncate with ellipsis
                "template_file_id": file_id,
                "similarity_score": round(similarity_score, 3),
                "download_url": f"/download_template_by_id/{file_id}" if file_id else None,
                "metadata": {
                    "category": metadata.get("category", "General"),
                    "upload_date": metadata.get("upload_date", None),
                    "file_type": metadata.get("file_type", "application/pdf")
                }
            }
            
            logger.info(f"Added template match: {title} (Score: {similarity_score:.3f})")
            matches.append(match)

        if not matches:
            logger.warning("No templates found above similarity threshold")
            # Return the closest match even if below threshold
            if documents and metadatas and distances:
                closest_idx = distances.index(min(distances))
                closest_doc = documents[closest_idx]
                closest_metadata = metadatas[closest_idx]
                closest_score = 1.0 - min(distances[closest_idx], 1.0)
                
                logger.info(f"Returning closest match with score: {closest_score:.3f}")
                return [{
                    "template_number": 1,
                    "template_title": closest_metadata.get("title", "Closest Match"),
                    "template_preview_text": closest_doc[:500] + "...",
                    "template_file_id": closest_metadata.get("file_id"),
                    "similarity_score": round(closest_score, 3),
                    "download_url": f"/download_template_by_id/{closest_metadata.get('file_id')}" if closest_metadata.get("file_id") else None,
                    "metadata": {
                        "category": closest_metadata.get("category", "General"),
                        "upload_date": closest_metadata.get("upload_date", None),
                        "file_type": closest_metadata.get("file_type", "application/pdf")
                    }
                }]
            else:
                return [{
                    "template_number": 1,
                    "template_title": "No Strong Match",
                    "template_preview_text": "No strong match found, but here's the closest resume template we have.",
                    "template_file_id": None,
                    "similarity_score": 0.0,
                    "download_url": None,
                    "metadata": {"category": "General"}
                }]

        return matches

    except Exception as e:
        logger.error(f"Error during template search: {e}")
        return [{
            "template_number": 1,
            "template_title": "Error",
            "template_preview_text": "An error occurred while searching for templates.",
            "template_file_id": None,
            "similarity_score": 0.0,
            "download_url": None,
            "metadata": {"category": "Error"}
        }]