import os
from enum import Enum
from pathlib import Path
from pydantic_settings import BaseSettings

APP_ROOT = Path(__file__).parents[2]

class AppMode(str, Enum):
	DEVELOPMENT = "development"
	PRODUCTION = "production"

def get_env() -> str:
	return os.getenv("APP_ENV", AppMode.DEVELOPMENT.value)

def load_env_file() -> Path:
	return APP_ROOT / f".env.{get_env()}"

class Settings(BaseSettings):
	# environment
	app_root: Path = APP_ROOT
	app_env: str = get_env()

	# Mistral
	mistral_api_key: str
	mistral_model_name: str
	mistral_model_embed_name: str

	# Groq
	groq_api_key: str
	groq_model_name: str

	# Openrouter
	openrouter_api_key: str
	openrouter_model_name: str
	openrouter_base_url: str

	# embeddings
	embedding_model_name: str

	# classifier
	classifier_model_dir: str
	classifier_label_binarizer_name: str
	classifier_model_name: str

	# vectordb
	vectordb_dir: str
	vectordb_host: str
	vectordb_port: int
	vectordb_collection_name: str

	# cache
	cachedb_dir: str
	cachedb_collection_name: str

	model_config = {
		"env_file": load_env_file(),
		"env_file_encoding": "utf-8",
		"extra": "ignore",
	}

settings = Settings()