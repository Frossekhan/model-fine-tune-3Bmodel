import json
import logging
from typing import Dict, List
from app.db.mongo_client import mongo_client
from app.config import settings

logger = logging.getLogger(__name__)


class MemoryService:
    SESSION_PREFIX = "assistant:session:"

    @classmethod
    def _get_filter(cls, session_id: str) -> Dict:
        return {"session_id": session_id}

    @classmethod
    def append_user_message(cls, session_id: str, content: str):
        document = {
            "session_id": session_id,
            "role": "user",
            "content": content,
            "timestamp": logging.Formatter("%Y-%m-%dT%H:%M:%S").formatTime(logging.LogRecord("", 0, "", 0, "", (), None)),
        }
        result = mongo_client.insert_one(document)
        if result:
            logger.debug("Appended user message to session %s", session_id)
        else:
            logger.warning("Failed to append user message to session %s", session_id)

    @classmethod
    def append_assistant_message(cls, session_id: str, content: str):
        document = {
            "session_id": session_id,
            "role": "assistant",
            "content": content,
            "timestamp": logging.Formatter("%Y-%m-%dT%H:%M:%S").formatTime(logging.LogRecord("", 0, "", 0, "", (), None)),
        }
        result = mongo_client.insert_one(document)
        if result:
            logger.debug("Appended assistant message to session %s", session_id)
        else:
            logger.warning("Failed to append assistant message to session %s", session_id)

    @classmethod
    def get_history(cls, session_id: str, limit: int = None) -> List[Dict[str, str]]:
        limit = limit or settings.max_history
        messages = mongo_client.find_many(
            filter_dict={"session_id": session_id},
            sort=[("timestamp", 1)],
            limit=limit,
        )
        # Convert MongoDB documents to simple message dicts
        result = []
        for msg in messages:
            result.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })
        return result

    @classmethod
    def clear_session(cls, session_id: str):
        deleted = mongo_client.delete_many({"session_id": session_id})
        logger.info("Cleared session memory %s, deleted %d messages", session_id, deleted)


memory_service = MemoryService()
