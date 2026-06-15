import logging
import redis.asyncio as redis
from app.config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    def __init__(self):
        self.redis = None

    async def connect(self):
        try:
            self.redis = redis.from_url(settings.redis_url, decode_responses=True)
            await self.redis.ping()
            logger.info("Connected to Redis memory store")
        except Exception as e:
            logger.warning("Redis connection failed: %s. Continuing without Redis.", str(e))
            self.redis = None

    async def disconnect(self):
        if self.redis:
            try:
                await self.redis.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.warning("Error closing Redis: %s", str(e))

    async def get(self, key: str):
        if not self.redis:
            return None
        return await self.redis.get(key)

    async def set(self, key: str, value: str, ex: int = None):
        if not self.redis:
            return None
        return await self.redis.set(key, value, ex=ex)

    async def lpush(self, key: str, *values):
        if not self.redis:
            return None
        return await self.redis.lpush(key, *values)

    async def lrange(self, key: str, start: int, end: int):
        if not self.redis:
            return []
        return await self.redis.lrange(key, start, end)

    async def delete(self, key: str):
        if not self.redis:
            return None
        return await self.redis.delete(key)

    async def expire(self, key: str, seconds: int):
        if not self.redis:
            return None
        return await self.redis.expire(key, seconds)


redis_client = RedisClient()
