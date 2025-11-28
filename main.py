from contextlib import asynccontextmanager
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from app.api.router import router
from app.config.logging_config import configure_logging
from app.services.service_registry import init_services, shutdown_services


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
	# Startup: initialize services sequentially
	await init_services()
	yield
	# Shutdown logic can be added here if needed
	await shutdown_services()


app = FastAPI(title="Jarvis LangChain Server", lifespan=lifespan)

origins = [
	"http://localhost:4000",
	"http://127.0.0.1:4000",
	"https://akshayrpatel.github.io",
]

app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

configure_logging()
app.include_router(router)


@app.get("/")
def root():
	return {"status": "jarvis-langchain-server running"}


@app.get("/health")
def health():
	return {"status": "healthy"}
