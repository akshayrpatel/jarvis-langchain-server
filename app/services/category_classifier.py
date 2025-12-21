import tracemalloc

tracemalloc.start()
snapshot1 = tracemalloc.take_snapshot()
print("Memory before loading classifier:", sum([stat.size for stat in snapshot1.statistics('filename')])/1024**2, "MB")

import os
import logging
import joblib
import numpy as np

from enum import Enum
from typing import List, Dict
from pydantic import BaseModel
from fastembed import TextEmbedding
from sklearn.preprocessing import MultiLabelBinarizer
from functools import lru_cache
from contextlib import contextmanager

from app.config.services import classifier_config

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["XGBOOST_NUM_THREADS"] = "1"
from xgboost import XGBClassifier


class Category(str, Enum):
	BACKGROUND = "background"
	CONTACT = "contact"
	EDUCATION = "education"
	EXPERIENCE = "experience"
	SKILLS = "skills"
	PROJECTS = "projects"
	PERSONAL = "personal"
	CALENDAR = "calendar"
	GENERAL = "general"

class CategoryInfo(BaseModel):
	name: Category
	description: str
	enable_rag: bool = True
	enable_tools: bool = False

CATEGORY_REGISTRY: Dict[Category, CategoryInfo] = {
	Category.BACKGROUND: CategoryInfo(
		name=Category.BACKGROUND,
		description="Personal background, biography, identity, life story, personal journey, "
            "origins, upbringing, personal history"
	),
	Category.CONTACT: CategoryInfo(
		name=Category.CONTACT,
		description="Contact information, ways to reach me, email, phone number, social media, "
            "communication details, messaging, address, how to get in touch",
	),
	Category.EDUCATION: CategoryInfo(
		name=Category.EDUCATION,
		description="Academic history, schooling, university, degrees, certifications, scholarships, "
            "courses, study experience, learning, educational achievements"
	),
	Category.EXPERIENCE: CategoryInfo(
		name=Category.EXPERIENCE,
		description="Professional experience, work history, roles, job responsibilities, employment, "
            "career achievements, companies, projects done at work, professional background"
	),
	Category.SKILLS: CategoryInfo(
		name=Category.SKILLS,
		description="Technical skills, expertise, tools, frameworks, programming languages, "
            "technologies, capabilities, knowledge areas, professional skills"
	),
	Category.PROJECTS: CategoryInfo(
		name=Category.PROJECTS,
		description="Projects, portfolio work, showcases, software applications, contributions, "
            "demonstrated work, personal or professional projects"
	),
	Category.PERSONAL: CategoryInfo(
		name=Category.PERSONAL,
		description="Casual chat, conversations, opinions, preferences, personality traits, hobbies, "
            "interests, personal anecdotes, informal interaction"
	),
	Category.CALENDAR: CategoryInfo(
		name=Category.CALENDAR,
		description="Scheduling, meetings, appointments, availability, calendar events, planning, "
            "dates, reminders, time management",
		enable_tools=True,
	),
	Category.GENERAL: CategoryInfo(
		name=Category.GENERAL,
		description="Default category for uncategorized queries, generic questions, miscellaneous topics, "
            "or when no other category matches",
		enable_rag=False,
		enable_tools=False,
	)
}


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedder(embed_model_name: str) -> TextEmbedding:
	return TextEmbedding(embed_model_name, threads=1)

@lru_cache(maxsize=1)
def get_label_binarizer(classifier_models_dir: str = classifier_config.models_dir,
                        model_label_binarizer_name: str = classifier_config.model_label_binarizer_name) -> MultiLabelBinarizer:
	path = f"{classifier_models_dir}/{model_label_binarizer_name}.pkl"
	return joblib.load(path)

@contextmanager
def get_model(classifier_models_dir: str, label: str):
	path = f"{classifier_models_dir}/xgb_{label}.pkl"
	model = joblib.load(path)
	try:
		yield model
	finally:
		del model

class CategoryClassifier:
	"""
	Classifies chatbot queries into one or more predefined categories.

  This classifier embeds known category descriptions at initialization and
  compares them to the embedding of an input query. Categories whose
  similarity score exceeds the configured threshold are returned.

  Attributes:
    embedder (TextEmbedding): Embedding model used to generate vector
        representations of both queries and category descriptions.
    threshold (float): Minimum similarity score required for a category
        to be considered a match.
  """

	def __init__(self,
	             model_name: str,
	             classifier_models_dir: str = classifier_config.models_dir,
	             threshold: float = 0.6):
		self.embed_model_name: str = model_name
		self.classifier_models_dir: str = classifier_models_dir
		self.threshold: float = threshold

	def _embed_text(self, text: str) -> np.ndarray:
		embedder: TextEmbedding = get_embedder(self.embed_model_name)
		return np.array(list(embedder.embed(text)))

	def classify(self, text: str) -> List[str]:
		categories: List[str] = []
		query_embedding: np.ndarray = self._embed_text(text)

		snapshot2 = tracemalloc.take_snapshot()
		print("Memory after embedding:", sum([stat.size for stat in snapshot2.statistics('filename')]) / 1024 ** 2, "MB")

		binarizer: MultiLabelBinarizer = get_label_binarizer(classifier_models_dir=self.classifier_models_dir)

		snapshot3 = tracemalloc.take_snapshot()
		print("Memory after binarizer:", sum([stat.size for stat in snapshot3.statistics('filename')]) / 1024 ** 2, "MB")

		for label in binarizer.classes_:
			with get_model(classifier_models_dir=self.classifier_models_dir, label=label) as model:
				prob: float = float(model.predict_proba(query_embedding)[0][1])
				snapshot4 = tracemalloc.take_snapshot()
				print("Memory after prediction:", sum([stat.size for stat in snapshot4.statistics('filename')]) / 1024 ** 2, "MB")

				if prob >= self.threshold:
					categories.append(label)

		if len(categories) == 0:
			categories.append(Category.GENERAL.value)

		logger.info("[CategoryClassifier] Categories for query: {}".format(categories))
		return categories

