# Autonomous Corporate Analyst 🤖📊

An autonomous, multi-agent research pipeline designed for corporate analysis. This project leverages a local FOSS (Free and Open-Source Software) AI stack to ensure data privacy and continuous, safe execution.

##  Tech Stack
* **Orchestration:** [LangGraph](https://python.langchain.com/docs/langgraph)
* **Local LLM:** Ollama (Running `Qwen2.5`)
* **Vector Store:** ChromaDB

##  Multi-Agent Architecture
The pipeline operates using a 4-agent autonomous system:
1. **Router Agent:** Analyzes the prompt and directs the workflow.
2. **RAG Agent:** Retrieves contextual information from the ChromaDB vector store.
3. **Writer Agent:** Drafts the corporate analysis based on retrieved context.
4. **Critic Agent:** Reviews the draft and triggers a "REVISE" feedback loop if it fails quality checks. 

*Note: The system includes a safety-first loop logic (`revision_count`) to prevent infinite generation loops.*

##  How to Run Locally
1. Ensure [Ollama](https://ollama.com/) is installed and running the `qwen2.5` model.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt