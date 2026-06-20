import logging
from typing import Any, Dict, List, Optional

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, PyMongoError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    MongoClient = None
    ConnectionFailure = Exception
    PyMongoError = Exception

from app.config import settings

logger = logging.getLogger(__name__)


class MongoClientWrapper:
    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db = None
        self.collection = None

    def connect(self) -> None:
        if not PYMONGO_AVAILABLE:
            logger.warning("pymongo is not installed. MongoDB features disabled. Install with: pip install pymongo")
            self.client = None
            self.db = None
            self.collection = None
            return
        try:
            self.client = MongoClient(
                settings.mongodb_url,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
            )
            # Test connection
            self.client.admin.command("ping")
            self.db = self.client[settings.mongodb_db]
            self.collection = self.db[settings.mongodb_collection]
            logger.info(
                "Connected to MongoDB at %s, db=%s, collection=%s",
                settings.mongodb_url,
                settings.mongodb_db,
                settings.mongodb_collection,
            )
        except ConnectionFailure as exc:
            logger.warning("MongoDB connection failed: %s. Continuing without MongoDB.", str(exc))
            self.client = None
            self.db = None
            self.collection = None
        except Exception as exc:
            logger.warning("MongoDB initialization error: %s", str(exc))
            self.client = None
            self.db = None
            self.collection = None

    def disconnect(self) -> None:
        if self.client:
            try:
                self.client.close()
                logger.info("MongoDB connection closed")
            except Exception as exc:
                logger.warning("Error closing MongoDB: %s", str(exc))
            self.client = None
            self.db = None
            self.collection = None

    def is_connected(self) -> bool:
        return self.client is not None and self.db is not None and self.collection is not None

    def insert_one(self, document: Dict[str, Any]) -> Optional[str]:
        if not self.is_connected():
            return None
        try:
            result = self.collection.insert_one(document)
            return str(result.inserted_id)
        except PyMongoError as exc:
            logger.warning("MongoDB insert_one failed: %s", str(exc))
            return None

    def find_one(self, filter_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.is_connected():
            return None
        try:
            return self.collection.find_one(filter_dict)
        except PyMongoError as exc:
            logger.warning("MongoDB find_one failed: %s", str(exc))
            return None

    def find_many(
        self,
        filter_dict: Dict[str, Any],
        sort: Optional[List[tuple]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.is_connected():
            return []
        try:
            cursor = self.collection.find(filter_dict)
            if sort:
                cursor = cursor.sort(sort)
            if limit:
                cursor = cursor.limit(limit)
            return list(cursor)
        except PyMongoError as exc:
            logger.warning("MongoDB find_many failed: %s", str(exc))
            return []

    def update_one(
        self,
        filter_dict: Dict[str, Any],
        update_dict: Dict[str, Any],
        upsert: bool = False,
    ) -> bool:
        if not self.is_connected():
            return False
        try:
            self.collection.update_one(filter_dict, update_dict, upsert=upsert)
            return True
        except PyMongoError as exc:
            logger.warning("MongoDB update_one failed: %s", str(exc))
            return False

    def delete_one(self, filter_dict: Dict[str, Any]) -> bool:
        if not self.is_connected():
            return False
        try:
            self.collection.delete_one(filter_dict)
            return True
        except PyMongoError as exc:
            logger.warning("MongoDB delete_one failed: %s", str(exc))
            return False

    def delete_many(self, filter_dict: Dict[str, Any]) -> int:
        if not self.is_connected():
            return 0
        try:
            result = self.collection.delete_many(filter_dict)
            return result.deleted_count
        except PyMongoError as exc:
            logger.warning("MongoDB delete_many failed: %s", str(exc))
            return 0

    def count_documents(self, filter_dict: Dict[str, Any]) -> int:
        if not self.is_connected():
            return 0
        try:
            return self.collection.count_documents(filter_dict)
        except PyMongoError as exc:
            logger.warning("MongoDB count_documents failed: %s", str(exc))
            return 0


mongo_client = MongoClientWrapper()