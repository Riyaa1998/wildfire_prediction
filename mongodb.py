import os
from pymongo import MongoClient
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBConnection:
    """Secure MongoDB connection handler with backward compatibility."""
    
    def __init__(self, username=None, password=None):
        """Initialize MongoDB connection with support for both old and new auth methods."""
        try:
            load_dotenv()  # Load environment variables
            if username and password:
                # Backward compatible with old code
                self.url = f"mongodb+srv://{username}:{password}@cluster0.bisvs.mongodb.net/?retryWrites=true&w=majority"
            else:
                # New secure way using environment variables
                self.url = os.getenv("MONGODB_URI")
                if not self.url:
                    raise ValueError("MONGODB_URI not found in environment variables")
            self.client = None
            logger.info("MongoDB connection initialized")
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            raise

    def getMongoClient(self):
        """Legacy method for backward compatibility."""
        return self.get_mongo_client()

    def get_mongo_client(self):
        """Create and return a MongoDB client."""
        try:
            if not self.client:
                self.client = MongoClient(
                    self.url,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=30000,
                    socketTimeoutMS=None,
                    connect=False
                )
                # Test the connection
                self.client.admin.command('ismaster')
                logger.info("Successfully connected to MongoDB")
            return self.client
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise

    def getDatabase(self, db_name):
        """Legacy method for backward compatibility."""
        return self.get_database(db_name)

    def get_database(self, db_name):
        """Get a database instance."""
        try:
            client = self.get_mongo_client()
            return client[db_name]
        except Exception as e:
            logger.error(f"Failed to access database {db_name}: {str(e)}")
            raise

    def getCollection(self, db_name, collection_name):
        """Legacy method for backward compatibility."""
        return self.get_collection(db_name, collection_name)

    def get_collection(self, db_name, collection_name):
        """Get a collection from the specified database."""
        try:
            database = self.get_database(db_name)
            return database[collection_name]
        except Exception as e:
            logger.error(f"Failed to access collection {collection_name}: {str(e)}")
            raise

    def getdata(self, db_name, collection_name):
        """Get all records from a collection (legacy method)."""
        try:
            collection = self.get_collection(db_name, collection_name)
            return list(collection.find())
        except Exception as e:
            logger.error(f"Failed to get data from {db_name}.{collection_name}: {str(e)}")
            raise

    def close_connection(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None
            logger.info("MongoDB connection closed")

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_connection()

# For backward compatibility
mongodbconnection = MongoDBConnection
