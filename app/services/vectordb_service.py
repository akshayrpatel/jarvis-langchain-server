import logging

from pathlib import Path
from typing import List
from chromadb import ClientAPI, PersistentClient, HttpClient
from chromadb.types import Collection

from app.config.services import vectordb_config, VectorDBMode

logger = logging.getLogger(__name__)


class VectorDBService:
	"""
	Service for connecting to and querying a ChromaDB vector database.

  Supports both:
    - Local persistent storage using ``PersistentClient``.
    - Remote ChromaDB servers using ``HttpClient``.

  Provides convenience methods for:
    - Creating/accessing collections.
    - Performing similarity search using embeddings.
    - Fetching documents filtered by assigned categories.
    - Closing PersistentClient resources safely.

  Attributes:
    mode (str): VectorDB mode (e.g. ``local`` or ``server``).
    persist_directory (Path): Directory used for persistent ChromaDB storage.
    host (str): Remote host for HTTP client mode.
    port (int): Remote port for HTTP client mode.
    ssl (bool): Whether SSL is enabled for remote communication.
    client (ClientAPI): The Active ChromaDB client instance.
    collection_name (str): Name of the collection to work with.
    collection (Collection): Collection instance for vector operations.
  """

	def __init__(self,
							 mode: str = VectorDBMode.LOCAL.value,
							 persist_directory: Path = vectordb_config.persist_directory,
							 host: str = 'localhost',
							 port: int = 8000,
							 ssl: bool = False,
							 collection_name: str = vectordb_config.collection_name):

		logger.info("[VectorDBService] Initializing (mode=%s, collection=%s)",mode, collection_name)
		self.mode: str = mode
		self.persist_directory: Path = persist_directory

		self.host: str = host
		self.port: int = port
		self.ssl: bool = ssl
		self.client: ClientAPI = self._create_client()

		self.collection_name: str = collection_name
		self.collection: Collection = self._create_collection(collection_name)
		logger.info("[VectorDBService] Initialized")

	def _create_client(self) -> ClientAPI:
		"""
    Create either a local ChromaDB client or a remote HTTP client.
    """

		if self.mode == VectorDBMode.LOCAL:
			logger.info("[VectorDBService] Using local persistent ChromaDB at %s", self.persist_directory)
			self.persist_directory.mkdir(parents=True, exist_ok=True)
			return PersistentClient(
				path=self.persist_directory
			)

		elif self.mode == VectorDBMode.SERVER:
			if not self.host or not self.port:
				logger.exception("[VectorDBService] Unable to connect to remote ChromaDB server without host and port")
				raise ValueError('host and port are required')

			logger.info(
				"[VectorDBService] Connecting to remote ChromaDB server: %s:%s (ssl=%s)",
				self.host, self.port, self.ssl
			)
			return HttpClient(host=self.host, port=self.port, ssl=self.ssl)

		else:
			logger.exception("[VectorDBService] Invalid vectordb mode: %s", self.mode)
			raise ValueError('invalid mode')

	def _create_collection(self, collection_name: str) -> Collection:
		"""
		Create collection if missing; otherwise returns existing one.
		"""

		try:
			logger.info("[VectorDBService] Using collection %s", collection_name)
			return self.client.get_or_create_collection(collection_name)
		except Exception:
			logger.exception("[VectorDBService] Failed to create or access collection")
			raise

	def similarity_search(self, query: str, embedding: List[float], k: int = 5) -> List[str]:
		"""
		Fetch top k relevant document chunks from vector db that are closest to the query embedding
		"""

		logger.debug("[VectorDBService] Running similarity search for query: %s", query)
		results = self.collection.query(
			query_embeddings=[embedding],
			n_results=k,
			include=["documents"]
		)

		if not results["documents"] or len(results["documents"][0]) == 0:
			return []

		logger.debug("[VectorDBService] Found %d documents for query", len(results["documents"][0]))
		return results["documents"][0]

	def similarity_search_by_category(self, query: str, categories: List[str]) -> List[str]:
		"""
		Fetch relevant document chunks from vector db that match the given categories
		"""

		logger.info("[VectorDBService] Running similarity search by category for query: %s", query[:10])
		results = self.collection.get(
		    where={"category": {"$in": categories}},
		    include=["documents", "metadatas"]
		)

		if len(results["documents"]) == 0:
			logger.info("[VectorDBService] Did not find any relevant documents for query: %s", query[:10])
			return []

		logger.info("[VectorDBService] Found %d documents for query", len(results["documents"]))
		return results["documents"]

	def close(self) -> None:
		"""
    Stops the underlying system for PersistentClient to release the
    SQLite connection/file lock.
    """
		logger.info("[VectorDBService] Closing connection and stopping system.")

		# 1. Check if the client is a PersistentClient (or has the necessary system component)
		if self.mode == VectorDBMode.LOCAL.value:
			# For PersistentClient, we must stop the internal system
			try:
				# Accessing internal methods is a known workaround for PersistentClient
				if hasattr(self.client, '_system') and hasattr(self.client._system, 'stop'):
					self.client._system.stop()
					logger.info("[VectorDBService] PersistentClient system stopped successfully.")
				else:
					logger.warning("[VectorDBService] Could not find internal stop method on PersistentClient.")
			except Exception as e:
				logger.error(f"[VectorDBService] Error stopping PersistentClient system: {e}")