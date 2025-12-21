import logging

from typing import List
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

from app.config.models import mistral_config, groq_config, openrouter_config

logger = logging.getLogger(__name__)


class LLMService:
	"""
	Service for querying multiple LLM providers with automatic failover.

  This service maintains a list of LLM providers, and attempts to
  query them in order until one successfully returns a response.

  Attributes:
    providers (List[BaseChatModel]): Ordered list of LLM providers. The
        service iterates through these providers in sequence and falls back
        to the next one if a provider fails.
  """

	def __init__(self) -> None:
		self.providers: List[BaseChatModel] = [
			ChatMistralAI(
				api_key=mistral_config.api_key,
				model_name=mistral_config.model,
				temperature=mistral_config.temperature,
				max_retries=mistral_config.max_retries
			),
			ChatOpenAI(
				api_key=openrouter_config.api_key,
				model=openrouter_config.model,
				base_url=openrouter_config.base_url,
				default_headers={
					"HTTP-Referer": "http://127.0.0.1:4000",
					"X-Title": "Jarvis Langchain Server"
				},
				temperature=openrouter_config.temperature,
			),
			ChatGroq(
				api_key=groq_config.api_key,
				model=groq_config.model,
				temperature=groq_config.temperature,
				max_retries=groq_config.max_retries,
			),
		]

	async def chat(self, messages: List[BaseMessage]) -> str:
		"""Send a list of langchain messages to llm."""
		for provider in self.providers:
			provider_class_name = type(provider)
			try:
				response = await provider.ainvoke(messages)
				logger.info(f"[LLMService] Successfully received response from {provider_class_name}")
				return response.content
				# return sample_response2
			except Exception as e:
				logger.warning(f"[LLMService] Failed to receive a response from {provider_class_name} : {e}")
				continue

		logger.error("[LLMService] All llm providers failed")
		return "Sorry, I am temporarily unavailable. Please try again later."