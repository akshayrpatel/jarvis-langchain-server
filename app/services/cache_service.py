import logging
import uuid

from pathlib import Path
from typing import List, Dict, Any, Set
from fastembed import TextEmbedding

from app.config.settings import settings
from app.config.services import CacheEvictionPolicy, cache_config
from app.services.category_classifier import Category
from app.services.vectordb_service import VectorDBService

logger = logging.getLogger(__name__)

CATEGORIES_TO_CACHE: Set[str] = {Category.BACKGROUND.value, Category.SKILLS.value, Category.EDUCATION.value,
                                 Category.EXPERIENCE.value, Category.CONTACT.value, Category.PERSONAL.value}

QUERY_MODEL_NAME: str = settings.embedding_model_name
QUERY_EMBEDDER: TextEmbedding = TextEmbedding(QUERY_MODEL_NAME, threads=1)

def embed_text(query: str) -> List[float]:
	return (list(QUERY_EMBEDDER.embed(query))[0]).tolist()


class CacheService(VectorDBService):
	"""
	{
		"documents": ["query"],
		"metadatas": [
			{
				"answer": "llm answer",
				"access_count": 2,
			}
		],
		"distances": [0.6]
	}

	"""

	def __init__(self,
	             persist_directory: Path = cache_config.persist_directory,
	             collection_name: str = cache_config.collection_name,
	             max_cache_size: int = cache_config.max_cache_size,
	             eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU,
	             threshold: float = cache_config.threshold,
	             **vectordb_kwargs):
		super().__init__(persist_directory=persist_directory, collection_name=collection_name, **vectordb_kwargs)
		self.max_cache_size = max_cache_size
		self.eviction_policy = eviction_policy
		self.threshold = threshold
		logger.info(f"[CacheService] Initialized (lazy) with "
		            f"collection_name: {self.collection_name}, "
		            f"max_cache_size: {self.max_cache_size}, "
		            f"eviction_policy: {self.eviction_policy}, "
		            f"threshold: {self.threshold}")

	def _evict_if_necessary(self) -> bool:

		count = self.collection.count()
		if count < self.max_cache_size:
			logger.info(f"[CacheService] Cache hasn't reached max size: {count} < {self.max_cache_size}")
			return False

		try:
			results = self.collection.get(
				include=["documents", "metadatas"]
			)

			metadatas = results["metadatas"]
			result_ids = results["ids"]

			total_results = len(metadatas)
			lfc = 2*31 - 1
			lf_id = None # id of the record to be removed

			for i in range(total_results):
				result_id = result_ids[i]
				metadata = metadatas[i]
				access_count = metadata["access_count"]
				if access_count < lfc:
					lf_id = result_id
					lfc = access_count

			logger.info(f"[CacheService] Deleting query with id: {lf_id}")
			self.collection.delete(ids=[lf_id])
			return True

		except Exception as e:
			logger.warning(f"[CacheService] Failed to evict an item from cache: {e}")
			return False

	def get(self, query: str) -> str | None:

		if self.client is None:
			logger.info(f"[CacheService] Initialize client")
			super().initialize_db_connection()

		try:
			embedding: List[float] = embed_text(query)
			results = self.collection.query(
				query_embeddings=[embedding],
				include=["metadatas", "distances"],
				n_results=1
			)

			if not results["metadatas"] or len(results["metadatas"][0]) == 0:
				logger.info("[CacheService] No similar queries in cache, i.e Cache MISS")
				return None

			result_distance: float = results["distances"][0][0]
			similarity: float = 1 - result_distance

			if similarity < self.threshold:
				logger.info("[CacheService] No similar queries above threshold, i.e Cache MISS")
				return None

			logger.info("[CacheService] Found similar query in cache, i.e Cache HIT")

			result_metadata: Dict[str, Any] = results["metadatas"][0][0]
			answer: str = result_metadata["answer"]
			result_doc_id: int = results["ids"][0][0]
			result_metadata["access_count"] = result_metadata.get("access_count", 0) + 1

			self.collection.update(
				ids=[result_doc_id],
				metadatas=[result_metadata]
			)
			return answer
		except Exception as e:
			logger.warning(f"[CacheService] CacheService failed to 'GET' with exception: {e}")
			return None

	def put(self, query: str, answer: str) -> None:

		if self.client is None:
			logger.info(f"[CacheService] Initialize client")
			super().initialize_db_connection()

		self._evict_if_necessary()

		try:
			doc_id: str = str(uuid.uuid4())
			embedding: List[float] = embed_text(query)
			metadata: Dict[str, Any] = {"answer": answer, "access_count": 0}

			self.collection.add(
				ids=[doc_id],
				documents=[query],
				embeddings=[embedding],
				metadatas=[metadata],
			)
		except Exception as e:
			logger.warning(f"[CacheService] CacheService failed to 'PUT' with exception: {e}")


