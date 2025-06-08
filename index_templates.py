from mongo_utils import get_mongodb_connection
from functions import extract_text_from_pdf
from rag_utils import add_template_to_vectorstore
from bson import ObjectId
import logging
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use persistent ChromaDB client
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

def index_templates():
    client, db, fs = get_mongodb_connection()
    
    if not client:
        logger.error("MongoDB connection failed.")
        return
    
    try:
        logger.info("Indexing templates from GridFS to ChromaDB...")
        
        # First, let's check what's actually in GridFS
        total_files = db.fs.files.count_documents({})
        logger.info(f"Total files in GridFS: {total_files}")
        
        if total_files == 0:
            logger.warning("No files found in GridFS. Make sure you've uploaded resume templates first.")
            return
        
        # Let's see what metadata exists
        sample_files = list(db.fs.files.find({}).limit(5))
        logger.info("Sample file metadata:")
        for file_doc in sample_files:
            logger.info(f"  File ID: {file_doc['_id']}, Filename: {file_doc.get('filename', 'N/A')}, Metadata: {file_doc.get('metadata', {})}")
        
        # Query all stored PDFs in GridFS (without the source filter first)
        files_cursor = db.fs.files.find({})
        
        count = 0
        for file_doc in files_cursor:
            try:
                file_id = file_doc["_id"]
                filename = file_doc.get("filename", f"Template_{count+1}")
                title = file_doc.get("metadata", {}).get("title", filename)
                
                logger.info(f"Processing file: {filename} (ID: {file_id})")
                
                # Get the file from GridFS
                try:
                    grid_out = fs.get(ObjectId(file_id))
                    pdf_bytes = grid_out.read()
                    
                    if len(pdf_bytes) == 0:
                        logger.warning(f"File {filename} is empty, skipping...")
                        continue
                    
                    logger.info(f"Successfully read {len(pdf_bytes)} bytes from {filename}")
                    
                except gridfs.NoFile:
                    logger.error(f"File with ID {file_id} not found in GridFS")
                    continue
                
                # Extract text from PDF
                try:
                    extracted_text = extract_text_from_pdf(pdf_bytes)
                    if not extracted_text or len(extracted_text.strip()) < 10:
                        logger.warning(f"No meaningful text extracted from {filename}, skipping...")
                        continue
                    
                    logger.info(f"Extracted {len(extracted_text)} characters from {filename}")
                    
                except Exception as e:
                    logger.error(f"Error extracting text from {filename}: {e}")
                    continue
                
                # Add to vector store
                try:
                    add_template_to_vectorstore(
                        title=str(file_id),  # Use file_id as unique identifier
                        content=extracted_text,
                        metadata={
                            "file_id": str(file_id),
                            "title": title,
                            "filename": filename
                        }
                    )
                    count += 1
                    logger.info(f"✅ Indexed: {title} ({file_id})")
                    
                except Exception as e:
                    logger.error(f"Error adding {filename} to vector store: {e}")
                    continue
                    
            except Exception as e:
                logger.error(f"❌ Error processing file {file_doc.get('_id')}: {e}")
        
        logger.info(f"✅ Finished indexing {count} templates from GridFS into ChromaDB.")
        
    except Exception as e:
        logger.error(f"Error during indexing process: {e}")
    finally:
        if client:
            client.close()

if __name__ == "__main__":
    index_templates()