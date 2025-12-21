from uuid import uuid4
from fastapi import APIRouter

from app.dto.chat_request import ChatRequest
from app.dto.chat_response import ChatResponse
from app.dto.rag_response import RAGResponse
from app.services.service_registry import service_registry

router = APIRouter(prefix="/api")

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
	session_id: str = request.session_id or str(uuid4())
	query: str = request.query

	rag_response: RAGResponse = await service_registry.rag.answer(
		query=query,
		session_id=session_id
	)

	return ChatResponse(
		answer=rag_response.markdown_text,
		session_id=session_id,
		followups=rag_response.followup_questions
	)

# @router.post("/chat", response_model=ChatResponse)
# async def chat(request: ChatRequest) -> ChatResponse:
# 	session_id: str = request.session_id or str(uuid4())
#
# 	rag_response = get_sample_rag_response()
# 	return ChatResponse(
# 		answer=rag_response.markdown_text,
# 		session_id=session_id,
# 		followups=rag_response.followup_questions
# 	)

def get_sample_rag_response() -> RAGResponse:
	return RAGResponse(
		markdown_text="**Good day!** I’m functioning at peak efficiency, as always—ready to assist with anything related to **Akshay Patel**.\n\nHere’s something you might find interesting:\nAkshay maintains **four key professional touchpoints** for collaboration, networking, and technical exploration:\n- **📧 Email** (`akshayrpatel24@gmail.com`): Preferred for *detailed inquiries* (job opportunities, project proposals, or technical discussions).\n- **💼 LinkedIn** ([linkedin.com/in/akshayrpatel](https://linkedin.com/in/akshayrpatel)): A hub for his *career trajectory*, certifications, and impact metrics—ideal for recruiters or peers in AI/software engineering.\n- **👨‍💻 GitHub** ([github.com/akshayrpatel](https://github.com/akshayrpatel)): Where his *coding style* and projects (from AI pipelines to web dev) live, complete with demos and documentation.\n- **🌐 Portfolio** ([akshayrpatel.github.io](https://akshayrpatel.github.io)): A curated showcase of *featured projects*, technologies, and interactive visualizations.\n\nHe actively updates these platforms and welcomes connections for **mentorship, freelance work, or speaking engagements** in tech/AI.\n\n*How might I assist you further?*",
		followup_questions=[
        "Which of Akshay’s projects (e.g., AI, web dev) would you like to explore in detail?",
        "Are you looking to connect with him for a *specific opportunity* (job, collaboration, mentorship)?",
        "Would you prefer a breakdown of his *technical skills* (e.g., Python, RAG pipelines) or *educational background*?",
        "How does Akshay’s approach to *open-source contributions* (via GitHub) align with your interests?",
        "Should I highlight his *preferred communication channels* for a particular type of inquiry?"
    ],
		response_quality="good"
	)