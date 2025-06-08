import os
import logging
from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_mongodb_connection():
    # Fixed: Use MONGODB_URI instead of MONGO_URI to match your .env file
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB_NAME", "resume_database")
    
    if not uri:
        logger.error("MONGODB_URI not found in environment variables")
        return None, None, None
    
    try:
        client = MongoClient(uri)
        db = client[db_name]
        fs = gridfs.GridFS(db)
        
        # Test the connection
        client.admin.command('ping')
        logger.info("Connected to MongoDB Atlas successfully")
        
        # Log database and collection info
        logger.info(f"Database: {db_name}")
        logger.info(f"Collections: {db.list_collection_names()}")
        
        return client, db, fs
    except Exception as e:
        logger.error(f"MongoDB Connection Error: {e}")
        return None, None, None