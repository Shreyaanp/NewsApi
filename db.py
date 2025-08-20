# app/db.py
import logging, asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne
from pymongo.errors import ServerSelectionTimeoutError
from .config import MONGO_URI, MONGO_DB

logger = logging.getLogger(__name__)
_client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5_000)
_db = _client[MONGO_DB]
_articles = _db.articles
_indexes_ready = False        # ← flag

async def _ensure_indexes():
    global _indexes_ready
    if _indexes_ready:
        return
    try:
        await _articles.create_index("url", unique=True)
        await _articles.create_index("cluster_id")
        _indexes_ready = True
    except ServerSelectionTimeoutError as e:
        logger.warning(f"MongoDB unavailable — persistence disabled: {e}")
        _indexes_ready = False

async def upsert_articles(articles):
    if not articles:
        return
    await _ensure_indexes()
    if not _indexes_ready:
        return                        # skip persistence if DB still down
    ops = [UpdateOne({"url": str(a.url)}, {"$set": a.model_dump()}, upsert=True)
           for a in articles]
    try:
        await _articles.bulk_write(ops, ordered=False)
    except Exception as e:
        logger.error(f"Mongo upsert failed: {e}")
