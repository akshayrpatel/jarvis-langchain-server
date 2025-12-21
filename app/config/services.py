from enum import Enum
from pathlib import Path
from pydantic import BaseModel

from app.config.settings import settings, AppMode


class ClassifierConfig(BaseModel):
	models_dir: Path = settings.app_root / settings.classifier_model_dir
	embedding_model_name: str = settings.embedding_model_name
	classifier_label_binarizer_name: str = settings.classifier_label_binarizer_name
	classifier_model_name: str = settings.classifier_model_name
	threshhold: float = 0.6

class MemoryConfig(BaseModel):
	history_length: int = 5

class RAGConfig(BaseModel):
	top_k: int = 10

class VectorDBMode(str, Enum):
	LOCAL = "local"
	SERVER = "server"

class VectorDBConfig(BaseModel):
	mode: str = VectorDBMode.SERVER.value if settings.app_env == AppMode.PRODUCTION.value else VectorDBMode.LOCAL.value
	persist_directory: Path = settings.app_root / settings.vectordb_dir
	host: str = settings.vectordb_host
	port: int = int(settings.vectordb_port)
	ssl: bool = True if settings.app_env == AppMode.PRODUCTION.value else False
	collection_name: str = settings.vectordb_collection_name

class CacheEvictionPolicy(str, Enum):
	LRU = "lru"

class CacheConfig(BaseModel):
	mode: str = VectorDBMode.SERVER.value if settings.app_env == AppMode.PRODUCTION.value else VectorDBMode.LOCAL.value
	model_local: str = settings.embedding_model_name
	persist_directory: Path = settings.app_root / settings.cachedb_dir
	collection_name: str = settings.cachedb_collection_name
	max_cache_size: int = 100
	eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU
	threshold: float = 0.9

classifier_config = ClassifierConfig()
memory_config = MemoryConfig()
rag_config = RAGConfig()
vectordb_config = VectorDBConfig()
cache_config = CacheConfig()
