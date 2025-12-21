from enum import Enum

from pydantic import BaseModel
from typing import List

class RAGResponseQuality(Enum):
	GOOD = 'good'
	BAD = 'bad'

class RAGResponse(BaseModel):
	markdown_text: str
	followup_questions: List[str]
	response_quality: str
