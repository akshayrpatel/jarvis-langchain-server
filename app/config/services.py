from enum import Enum
from pathlib import Path
from pydantic import BaseModel

from app.config.settings import settings, AppMode


class ClassifierConfig(BaseModel):
	model_local: str = settings.embedding_model_local_name
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

classifier_config = ClassifierConfig()
memory_config = MemoryConfig()
rag_config = RAGConfig()
vectordb_config = VectorDBConfig()
