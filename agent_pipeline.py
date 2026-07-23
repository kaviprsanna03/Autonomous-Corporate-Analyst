# --- 1. ALL IMPORTS ---
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama, OllamaLLM
from langgraph.graph import StateGraph, END

# --- 2. DATA MODELS & STATE ---
class RetrievedDocument(BaseModel):
    content: str
    source_file: str

class FinalReport(BaseModel):
    summary: str
    key_takeaways: List[str]

class RouterOutput(BaseModel):
    sub_tasks: List[str] = Field(description="List of step-by-step tasks to research the user's query")

# ⚠️ Missing Class Added Here
class CriticOutput(BaseModel):
    decision: str = Field(description="Must be exactly 'ACCEPT' or 'REVISE'")
    feedback: str = Field(description="Detailed feedback on what is missing or needs improvement based on the original query")

class GraphState(TypedDict):
    """This is the state object passed between all our agents."""
    user_query: str                  
    sub_tasks: List[str]             
    retrieved_docs: List[RetrievedDocument] 
    current_draft: str               
    critic_feedback: str             
    revision_count: int              
    decision: str             

# --- 3. GLOBAL SETUP ---
chat_llm = ChatOllama(model="qwen2.5:3b", temperature=0)
writer_llm = OllamaLLM(model="qwen2.5:3b", temperature=0.3) 
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- 4. DATABASE SETUP FUNCTION ---
def setup_vector_database(pdf_path: str):
    """Loads a PDF, chunks it, and saves it to a local Chroma database."""
    print(f"---  LOADING DOCUMENT: {pdf_path} ---")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    print("---  STORING IN CHROMADB ---")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    return vectorstore

# --- 5. AGENT FUNCTIONS ---

# AGENT 1: ROUTER
def router_node(state: GraphState):
    print("---  ROUTER AGENT: Thinking ---")
    query = state["user_query"]
    prompt = f"You are a research manager. Break down this complex query into 2 or 3 specific research tasks: {query}"
    
    structured_llm = chat_llm.with_structured_output(RouterOutput)
    response = structured_llm.invoke(prompt)
    
    return {"sub_tasks": response.sub_tasks}

# AGENT 2: RAG SPECIALIST
def rag_node(state: GraphState):
    print("---  RAG SPECIALIST: Searching Database ---")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    search_query = state["sub_tasks"][0]
    print(f"Executing Search Query: {search_query[:50]}...")
    
    docs = retriever.invoke(search_query)
    
    retrieved_data = []
    for doc in docs:
        retrieved_data.append(
            RetrievedDocument(
                content=doc.page_content, 
                source_file=doc.metadata.get("source", "Unknown")
            )
        )
    return {"retrieved_docs": retrieved_data}

# AGENT 3: THE WRITER
def writer_node(state: GraphState):
    print("---  WRITER AGENT: Drafting the Report ---")
    tasks = state.get("sub_tasks", [])
    docs = state.get("retrieved_docs", [])
    
    tasks_text = "\n".join(tasks)
    context_text = "\n\n".join([f"Document Snippet:\n{doc.content}" for doc in docs])
    
    writer_prompt = PromptTemplate(
        template="""You are an elite corporate research analyst. 
        Write a professional, well-structured financial draft addressing the specific tasks below.
        Base your analysis STRICTLY on the provided Context. Do not hallucinate or invent numbers.
        
        Tasks to address:
        {tasks}
        
        Context Data:
        {context}
        
        Write the detailed draft below:
        """,
        input_variables=["tasks", "context"]
    )
    
    chain = writer_prompt | writer_llm
    print(" Writer is synthesizing data... (This might take a minute)")
    draft = chain.invoke({"tasks": tasks_text, "context": context_text})
    
    return {"current_draft": draft}

# AGENT 4: THE CRITIC
def critic_node(state: GraphState):
    print("---  CRITIC AGENT: Reviewing the Draft ---")
    query = state["user_query"]
    draft = state["current_draft"]
    revision_count = state.get("revision_count", 0)
    
    if revision_count >= 2:
        print(" Max revisions reached. Accepting current draft to avoid infinite loop.")
        return {"decision": "ACCEPT", "critic_feedback": "Max revisions reached.", "revision_count": revision_count}
    
    prompt = f"""You are a strict Managing Director reviewing a financial report draft.
    Original User Query: {query}
    
    Draft Report:
    {draft}
    
    Does this draft fully and accurately answer the user's query? Specifically, check if requested numerical data (like net profit) or specific details are present.
    If the draft perfectly answers the query, output decision 'ACCEPT'.
    If the draft is missing crucial data requested in the query (like specific numbers), output decision 'REVISE' and provide specific feedback on what is missing.
    """
    
    structured_llm = chat_llm.with_structured_output(CriticOutput)
    print(" Critic is analyzing the draft...")
    response = structured_llm.invoke(prompt)
    
    print(f"\n[Critic Decision]: {response.decision}")
    print(f"[Critic Feedback]: {response.feedback}\n")
    
    return {
        "decision": response.decision,
        "critic_feedback": response.feedback,
        "revision_count": revision_count + 1
    }

# --- 6. LANGGRAPH ORCHESTRATION ---
def build_workflow():
    print(" Building LangGraph Workflow...")
    workflow = StateGraph(GraphState)
    
    # 1. Add all our agents as "Nodes"
    workflow.add_node("Router", router_node)
    workflow.add_node("RAG_Specialist", rag_node)
    workflow.add_node("Writer", writer_node)
    workflow.add_node("Critic", critic_node)
    
    # 2. Connect the standard flow (Edges)
    workflow.set_entry_point("Router")
    workflow.add_edge("Router", "RAG_Specialist")
    workflow.add_edge("RAG_Specialist", "Writer")
    workflow.add_edge("Writer", "Critic")
    
    # 3. The Magic: Conditional Routing
    def decide_next_step(state: GraphState):
        decision = state.get("decision", "ACCEPT")
        if decision == "REVISE":
            print(" ROUTING: Critic requested revision. Sending back to Writer...")
            return "revise"
        else:
            print(" ROUTING: Critic accepted. Finishing up...")
            return "end"

    workflow.add_conditional_edges(
        "Critic",
        decide_next_step,
        {
            "revise": "Writer", # Loop back to Writer
            "end": END          # Stop the graph
        }
    )
    
    # Compile the graph into a runnable application
    return workflow.compile()

# --- 7. MAIN TEST BLOCK ---
if __name__ == "__main__":
    print(" Starting Autonomous LangGraph Pipeline...")
    
    app = build_workflow()
    
    # Initial state
    initial_state = {
        "user_query": "Analyze Tesla's Q3 performance focusing on supply chain and net profit",
        "sub_tasks": [],
        "retrieved_docs": [],
        "current_draft": "",
        "critic_feedback": "",
        "revision_count": 0,
        "decision": ""
    }
    
    # Run the graph autonomously!
    print("\n RUNNING THE AUTONOMOUS LOOP \n")
    final_state = app.invoke(initial_state)
    
    print("\n=========================================")
    print(" PIPELINE FINISHED SUCCESSFULLY!")
    print(f"Total Revisions Done: {final_state['revision_count']}")
    print("Final Decision:", final_state['decision'])
    print("=========================================")