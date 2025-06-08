from mongo_utils import get_mongodb_connection
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_mongodb():
    client, db, fs = get_mongodb_connection()
    
    if not client:
        logger.error("Cannot connect to MongoDB")
        return
    
    try:
        # Check database info
        logger.info(f"Database: {db.name}")
        logger.info(f"Collections: {db.list_collection_names()}")
        
        # Check GridFS
        logger.info("\n=== GridFS Debug Info ===")
        total_files = db.fs.files.count_documents({})
        total_chunks = db.fs.chunks.count_documents({})
        
        logger.info(f"Total files in fs.files: {total_files}")
        logger.info(f"Total chunks in fs.chunks: {total_chunks}")
        
        if total_files > 0:
            logger.info("\n=== Sample Files ===")
            for i, file_doc in enumerate(db.fs.files.find({}).limit(5)):
                logger.info(f"File {i+1}:")
                logger.info(f"  ID: {file_doc['_id']}")
                logger.info(f"  Filename: {file_doc.get('filename', 'N/A')}")
                logger.info(f"  Length: {file_doc.get('length', 'N/A')} bytes")
                logger.info(f"  Upload Date: {file_doc.get('uploadDate', 'N/A')}")
                logger.info(f"  Metadata: {file_doc.get('metadata', {})}")
                logger.info("  ---")
        
        # Check regular collections
        logger.info(f"\n=== Regular Collections ===")
        resumes_collection = db.get_collection("resumes")
        resume_count = resumes_collection.count_documents({})
        logger.info(f"Documents in 'resumes' collection: {resume_count}")
        
        if resume_count > 0:
            sample_resume = resumes_collection.find_one({})
            logger.info(f"Sample resume document keys: {list(sample_resume.keys()) if sample_resume else 'None'}")
        
        # Check for any collection that might contain resume templates
        for collection_name in db.list_collection_names():
            if collection_name not in ['fs.files', 'fs.chunks']:
                count = db.get_collection(collection_name).count_documents({})
                logger.info(f"Collection '{collection_name}': {count} documents")
        
    except Exception as e:
        logger.error(f"Error during debugging: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    debug_mongodb()