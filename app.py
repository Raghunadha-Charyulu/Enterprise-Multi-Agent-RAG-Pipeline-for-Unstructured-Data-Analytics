import streamlit as st
import os
import chromadb
from typing import TypedDict, Dict, Any
from pypdf import PdfReader

# Frameworks
from llama_index.core.node_parser import SentenceSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END

# Google Gemini Setup
import google.generativeai as genai

# 1. STREAMLIT UI SETUP
st.set_page_config(page_title="Enterprise Multi-Agent RAG", layout="wide")
st.title("🤖 Enterprise Multi-Agent RAG Pipeline")
st.caption("Powered by LangGraph, LlamaIndex, ChromaDB & Gemini API")

# Sidebar for API Key configuration
with st.sidebar:
    st.header("Configuration")
    gemini_key = st.text_input("Enter Google Gemini API Key:", type="password")
    if gemini_key:
        genai.configure(api_key=gemini_key)
        os.environ["GEMINI_API_KEY"] = gemini_key

# 2. DEFINE THE LANGGRAPH AGENT STATE
class AgentState(TypedDict):
    query: str
    context: str
    documents: list
    relevance_score: str
    response: str

# 3. LOCAL VECTOR STORE ORCHESTRATION (Data Engineering Layer)
@st.cache_resource
def get_vector_db():
    # Initialize local persistent ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="enterprise_knowledge")
    # Load free local SBERT embedding model
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return collection, embed_model

collection, embed_model = get_vector_db()

# File Upload Processing Layout
uploaded_file = st.file_uploader("Upload an Enterprise Document (PDF)", type=["pdf"])

if uploaded_file and gemini_key:
    if st.button("Ingest and Process Document"):
        with st.spinner("LlamaIndex parsing and chunking document..."):
            # Read PDF lines safely
            reader = PdfReader(uploaded_file)
            raw_text = ""
            for page in reader.pages:
                raw_text += page.extract_text() or ""
            
            # Using LlamaIndex's advanced structural chunking strategy
            parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
            chunks = parser.split_text(raw_text)
            
            # Embed chunks and upsert data records into ChromaDB
            for i, chunk in enumerate(chunks):
                vector = embed_model.embed_query(chunk)
                collection.upsert(
                    ids=[f"doc_{uploaded_file.name}_{i}"],
                    embeddings=[vector],
                    documents=[chunk],
                    metadatas=[{"source": uploaded_file.name}]
                )
        st.success(f"Successfully chunked into {len(chunks)} records and stored in Vector Database!")

# 4. DEFINE THE MULTI-AGENT STATE NODES
def retrieval_agent(state: AgentState) -> Dict[str, Any]:
    """Agent 1: Mathematical Cosine Similarity Search Node"""
    query_vector = embed_model.embed_query(state["query"])
    # Query ChromaDB collection for the top 3 most relevant context matches
    results = collection.query(query_embeddings=[query_vector], n_results=3)
    docs = results['documents'][0] if results['documents'] else []
    return {"documents": docs, "context": "\n".join(docs)}

def evaluation_agent(state: AgentState) -> Dict[str, Any]:
    """Agent 2: Context Relevancy & Hallucination Guard Node"""
    if not state["documents"]:
        return {"relevance_score": "NO"}
    
    # Use Gemini API to validate if context matches query intent structurally
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"""
    Analyze if the given background context contains direct analytical info to answer the query.
    Query: {state['query']}
    Context: {state['context']}
    Reply with ONLY one single word: 'YES' or 'NO'.
    """
    try:
        response = model.generate_content(prompt)
        score = response.text.strip().upper()
        return {"relevance_score": "YES" if "YES" in score else "NO"}
    except Exception:
        return {"relevance_score": "YES"} # Fallback option

def generation_agent(state: AgentState) -> Dict[str, Any]:
    """Agent 3: Synthesis & Final Answer Production Node"""
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"""
    You are an expert Enterprise AI Agent. Answer the user query using only the validated context provided.
    Query: {state['query']}
    Context: {state['context']}
    Answer:
    """
    response = model.generate_content(prompt)
    return {"response": response.text}

# 5. CONSTRUCT THE LANGGRAPH WORKFLOW FLOW CHART
workflow = StateGraph(AgentState)

# Add our independent specialized execution agents
workflow.add_node("RetrieveData", retrieval_agent)
workflow.add_node("EvaluateRelevance", evaluation_agent)
workflow.add_node("GenerateAnswer", generation_agent)

# Set up orchestration pipeline connections
workflow.set_entry_point("RetrieveData")
workflow.add_edge("RetrieveData", "EvaluateRelevance")

# Route based on the Evaluation Agent's grading decision
workflow.add_conditional_edges(
    "EvaluateRelevance",
    lambda state: "continue" if state["relevance_score"] == "YES" else "end",
    {
        "continue": "GenerateAnswer",
        "end": END
    }
)
workflow.add_edge("GenerateAnswer", END)
app_compiled = workflow.compile()

# 6. USER QUERYING AND WEB VIEW RUNNER
user_query = st.text_input("Ask a question regarding your ingested unstructured data assets:")

if user_query:
    if not gemini_key:
        st.warning("Please provide a valid Gemini API Key to activate the evaluation and generation agents.")
    else:
        with st.spinner("Executing State Graph Pipeline Workflows..."):
            # Stream the query through the state graph engine
            inputs = {"query": user_query, "documents": [], "context": "", "relevance_score": "", "response": ""}
            final_output = app_compiled.invoke(inputs)
            
            # Render outputs onto web view dashboard panels
            st.subheader("Final Agent Output:")
            if final_output.get("response"):
                st.write(final_output["response"])
            else:
                st.error("The evaluation agent blocked generation because the context was deemed irrelevant to avoid hallucination.")
            
            with st.expander("View Agent Workspace Ingestion Logs"):
                st.write("**Evaluation Agent Score:**", final_output.get("relevance_score"))
                st.write("**Retrieved Core Documents Used:**", final_output.get("documents"))