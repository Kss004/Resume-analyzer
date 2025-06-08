import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_chroma():
    try:
        # Initialize ChromaDB with persistent storage
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Get the collection
        collection = chroma_client.get_collection(
            name="resume_templates",
            embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-3-small"
            )
        )
        
        # Get collection info
        count = collection.count()
        logger.info(f"Total documents in collection: {count}")
        
        # Get all documents with metadata
        results = collection.get()
        
        if results and results['ids']:
            logger.info("\nSample documents in collection:")
            for i, (doc_id, metadata) in enumerate(zip(results['ids'][:5], results['metadatas'][:5])):
                logger.info(f"\nDocument {i+1}:")
                logger.info(f"ID: {doc_id}")
                logger.info(f"Metadata: {metadata}")
                logger.info(f"Content preview: {results['documents'][i][:100]}...")
        else:
            logger.warning("No documents found in collection!")
            
    except Exception as e:
        logger.error(f"Error debugging ChromaDB: {e}")

if __name__ == "__main__":
    debug_chroma() 