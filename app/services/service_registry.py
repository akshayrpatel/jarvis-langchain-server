import logging

from app.config.services import vectordb_config, cache_config
from app.services.cache_service import CacheService
from app.services.llm_service import LLMService
from app.services.memory_service import MemoryService
from app.services.rag_service import RAGService
from app.services.vectordb_service import VectorDBService

logger = logging.getLogger(__name__)


class ServiceRegistry:
	"""
	Registry for globally accessible service instances.

  This registry holds singletons of core application services, including:
    - ``LLMService`` for interacting with language models
    - ``MemoryService`` for storing chat history
    - ``VectorDBService`` for similarity search and RAG retrieval
    - ``RAGService`` for the full RAG pipeline

  Instances are created during application startup by ``init_services`` and
  are available via the global ``service_registry`` instance.

  Attributes:
    llm (LLMService | None): Singleton LLM service.
    memory (MemoryService | None): Singleton conversation memory service.
    vectordb (VectorDBService | None): Singleton vector database service.
    rag (RAGService | None): Singleton RAG pipeline service.
  """

	llm: LLMService | None = None
	memory: MemoryService | None = None
	vectordb: VectorDBService | None = None
	rag: RAGService | None = None
	cache: CacheService | None = None

service_registry = ServiceRegistry()

async def init_services() -> None:
	logger.info("[ServiceRegistry] Initializing services")
	service_registry.llm = LLMService()
	service_registry.memory = MemoryService()
	service_registry.vectordb = VectorDBService(
		mode=vectordb_config.mode,
		host=vectordb_config.host,
		port=vectordb_config.port,
		ssl=vectordb_config.ssl
	)
	service_registry.cache = CacheService(
		collection_name=cache_config.collection_name
	)
	service_registry.rag = RAGService(
		memory=service_registry.memory,
		vectordb=service_registry.vectordb,
		llm=service_registry.llm,
		cache=service_registry.cache
	)
	logger.info("[ServiceRegistry] Initialized")

async def shutdown_services() -> None:
	logger.info("[ServiceRegistry] Shutting down services")
	service_registry.vectordb.close()
	pass


