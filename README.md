# **Jarvis LangChain Server**

Jarvis LangChain Server is the backend to my personal AI-powered portfolio assistant, built to experiment with modern AI frameworks and allow visitors to interactively explore my work and projects.


## **Motivation & Purpose**

I wanted to create a system where visitors to my [portfolio](https://akshayrpatel.github.io/) website could **ask questions about me** and get intelligent, context-aware responses. Beyond being a portfolio feature, this project allowed me to **experiment with the latest AI technologies** and understand how real-world LLM-powered systems are designed and orchestrated.

The purpose of this project was to **learn and integrate state-of-the-art AI and ML frameworks**, while also building a **functional RAG-based assistant** for my portfolio.

## **Components**

The system is designed with modular services, each with a clear responsibility:

* **MemoryService** – Stores conversation history per session, enabling context-aware responses
* **CategoryClassifier** – Classifies queries into categories, allowing targeted retrieval from the knowledge base
* **VectorDBService** – Interfaces with ChromaDB to perform semantic similarity searches over document embeddings
* **LLMService** – Wraps multiple LLM providers (Mistral, OpenAI/OpenRouter, Groq) with automatic failover
* **RAGService** – Orchestrates retrieval, memory, and LLM invocation to produce final answers
* **ServiceRegistry** – Centralizes initialization and lifecycle management of all services

Together, these components form a **retrieval-augmented generation (RAG) pipeline**. 

## **Flow**

When a user asks a question, the system:

1. Stores the query in memory
2. Classifies the query into categories
3. Retrieves relevant documents using embeddings from the vector database
4. Formats the query with context using prompt templates
5. Sends it to the LLM to generate a coherent, context-aware response

This design ensures **relevant, accurate, and contextual answers** while allowing me to experiment with LLM integration, memory management, and retrieval systems.


## **Architecture**

The architecture is modular and service-oriented, designed for clarity and maintainability.

* Each component is **self-contained** and testable
* Services communicate through the **ServiceRegistry**, avoiding circular dependencies
* The **RAG pipeline** orchestrates retrieval, memory, and generation

*A diagram will be added here later to visually depict the system.*


## **Technologies and Concepts**

* **FastAPI** – Modern Python web framework for building async APIs
* **Uvicorn** – ASGI server to run FastAPI apps efficiently
* **LangChain** – For memory management, message orchestration, and chat abstractions
* **ChromaDB** – Vector database for semantic document retrieval using embeddings
* **Large Language Models (LLMs)** – Mistral, OpenAI/OpenRouter, Groq for text generation
* **RAG (Retrieval-Augmented Generation)** – Combines retrieval from documents with LLM generation for accurate, context-aware answers
* **Async Python** – For handling multiple LLM providers concurrently
* **Prompt Templates & Engineering** – Dynamic templates to structure context for LLMs
* **Python Typing and Docstrings** – For maintainability, clarity, and clean code

The project combines **state-of-the-art AI concepts** with **practical software engineering principles**, like modularity, service orchestration, and failover strategies.


## **How I Built It and Challenges Faced**

I built this project as part of my self-learning journey:

* I took courses on **LangChain, RAG systems, and vector search** to understand their practical applications
* I designed **modular Python services** to separate concerns and make the system testable
* Implemented **category-aware retrieval** to improve relevance and reduce irrelevant results
* Integrated multiple **LLM providers** to ensure reliability and failover in case of errors
* Faced challenges with **hosting, async operations, prompts, and vector database integration**
* Learned how to design **end-to-end LLM pipelines** that are maintainable and extensible

Despite the challenges, I successfully built a working system that demonstrates **how RAG and LLMs can power a personal AI assistant**.


## **Key Learnings**

This project taught me both **conceptual and practical skills**:

* **RAG pipelines** – How to integrate retrieval systems with LLMs for context-aware responses
* **Vector search and embeddings** – How to represent and search semantic information efficiently
* **LLM orchestration** – Handling multiple providers, failover strategies, and asynchronous queries
* **Memory management** – Maintaining session history to improve conversational flow
* **Prompt engineering & template design** – Structuring context for coherent AI responses
* **Modular service design** – Building maintainable, testable, and extendable systems
* **Debugging and hosting** – Solving real-world challenges in multi-component AI applications
