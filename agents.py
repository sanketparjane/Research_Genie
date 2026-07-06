import os
import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from .state import ResearchState
from .tools import search_arxiv, download_pdf, extract_text_from_pdf
from .rag import RAGManager
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = ChatGroq(temperature=0.2, model_name="llama3-70b-8192")
rag_manager = RAGManager()

def planner_agent(state: ResearchState) -> ResearchState:
    print("--- PLANNER AGENT ---")
    query = state.get("query", "")
    
    prompt = f"""You are a research planning agent. Given the following research query, break it down into 3-5 specific, searchable subtasks or keywords to find the most relevant academic papers on ArXiv.
    Return ONLY a JSON list of strings representing the subtasks/keywords. Do not include markdown formatting like ```json.
    
    Query: {query}"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        subtasks = json.loads(response.content.strip())
    except json.JSONDecodeError:
        # Fallback if json parsing fails
        subtasks = [query]
        
    return {"subtasks": subtasks, "current_step": "Planner Agent finished."}

def search_agent(state: ResearchState) -> ResearchState:
    print("--- SEARCH AGENT ---")
    subtasks = state.get("subtasks", [])
    
    all_results = []
    # Search for the first couple of subtasks to avoid too many results
    for task in subtasks[:2]:
        results = search_arxiv(task, max_results=2)
        all_results.extend(results)
        
    # Deduplicate based on entry_id
    seen = set()
    unique_results = []
    for r in all_results:
        if r["entry_id"] not in seen:
            seen.add(r["entry_id"])
            unique_results.append(r)
            
    return {"search_results": unique_results, "current_step": "Search Agent finished."}

def pdf_agent(state: ResearchState) -> ResearchState:
    print("--- PDF AGENT ---")
    search_results = state.get("search_results", [])
    
    downloaded_pdfs = []
    extracted_texts = {}
    
    for paper in search_results:
        pdf_url = paper.get("pdf_url")
        entry_id = paper.get("entry_id")
        title = paper.get("title")
        
        if pdf_url:
            pdf_path = download_pdf(pdf_url)
            downloaded_pdfs.append(pdf_path)
            
            # Extract text
            text = extract_text_from_pdf(pdf_path)
            extracted_texts[title] = text
            
            # Add to RAG
            rag_manager.add_texts(text, metadata={"title": title, "entry_id": entry_id})
            
    return {
        "downloaded_pdfs": downloaded_pdfs, 
        "extracted_texts": extracted_texts,
        "current_step": "PDF Agent finished."
    }

def rag_agent(state: ResearchState) -> ResearchState:
    print("--- RAG AGENT ---")
    query = state.get("query", "")
    
    # Retrieve relevant contexts
    docs = rag_manager.retrieve(query, k=10)
    context = "\n\n".join([f"Source: {doc.metadata.get('title', 'Unknown')}\n{doc.page_content}" for doc in docs])
    
    return {"rag_context": context, "current_step": "RAG Agent finished."}

def summarization_agent(state: ResearchState) -> ResearchState:
    print("--- SUMMARIZATION AGENT ---")
    extracted_texts = state.get("extracted_texts", {})
    
    summaries = {}
    for title, text in extracted_texts.items():
        # Truncate text to avoid exceeding token limits
        truncated_text = text[:15000] 
        prompt = f"Summarize the following research paper concisely in 2-3 paragraphs. Highlight the main contributions and methodology.\n\nTitle: {title}\nText: {truncated_text}"
        response = llm.invoke([HumanMessage(content=prompt)])
        summaries[title] = response.content
        
    return {"summaries": summaries, "current_step": "Summarization Agent finished."}

def comparison_agent(state: ResearchState) -> ResearchState:
    print("--- COMPARISON AGENT ---")
    summaries = state.get("summaries", {})
    
    if not summaries:
        return {"comparison": "No papers to compare.", "current_step": "Comparison Agent finished."}
        
    prompt = "Compare and contrast the following research papers based on their summaries. Highlight their different approaches, advantages, and disadvantages.\n\n"
    for title, summary in summaries.items():
        prompt += f"Title: {title}\nSummary: {summary}\n\n"
        
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"comparison": response.content, "current_step": "Comparison Agent finished."}

def citation_agent(state: ResearchState) -> ResearchState:
    print("--- CITATION AGENT ---")
    search_results = state.get("search_results", [])
    
    if not search_results:
         return {"citations": "No citations available.", "current_step": "Citation Agent finished."}
         
    prompt = "Generate APA, IEEE, and BibTeX citations for the following papers:\n\n"
    for paper in search_results:
        prompt += f"Title: {paper.get('title')}\nAuthors: {', '.join(paper.get('authors', []))}\nPublished: {paper.get('published')}\nURL: {paper.get('entry_id')}\n\n"
        
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"citations": response.content, "current_step": "Citation Agent finished."}

def research_gap_agent(state: ResearchState) -> ResearchState:
    print("--- RESEARCH GAP AGENT ---")
    comparison = state.get("comparison", "")
    rag_context = state.get("rag_context", "")
    
    prompt = f"""Based on the following comparison of recent papers and the retrieved context, identify 3-5 unexplored research directions or "research gaps".
    
    Comparison: {comparison}
    
    Additional Context: {rag_context[:5000]}"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"research_gaps": response.content, "current_step": "Research Gap Agent finished."}

def compile_report(state: ResearchState) -> ResearchState:
    print("--- COMPILING REPORT ---")
    
    report = f"# Comprehensive Literature Review\n\n"
    
    report += "## Executive Summaries\n\n"
    for title, summary in state.get("summaries", {}).items():
        report += f"### {title}\n{summary}\n\n"
        
    report += "## Comparative Analysis\n\n"
    report += state.get("comparison", "") + "\n\n"
    
    report += "## Identified Research Gaps\n\n"
    report += state.get("research_gaps", "") + "\n\n"
    
    report += "## Citations\n\n"
    report += state.get("citations", "") + "\n\n"
    
    return {"final_report": report, "current_step": "Workflow completed."}
