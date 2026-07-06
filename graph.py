from langgraph.graph import StateGraph, END
from .state import ResearchState
from .agents import (
    planner_agent, search_agent, pdf_agent, rag_agent, 
    summarization_agent, comparison_agent, citation_agent, 
    research_gap_agent, compile_report
)

def create_research_graph():
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("planner", planner_agent)
    workflow.add_node("search", search_agent)
    workflow.add_node("pdf", pdf_agent)
    workflow.add_node("rag", rag_agent)
    workflow.add_node("summarization", summarization_agent)
    workflow.add_node("comparison", comparison_agent)
    workflow.add_node("citation", citation_agent)
    workflow.add_node("research_gap", research_gap_agent)
    workflow.add_node("compile_report", compile_report)
    
    # Define edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "search")
    workflow.add_edge("search", "pdf")
    workflow.add_edge("pdf", "rag")
    workflow.add_edge("rag", "summarization")
    workflow.add_edge("summarization", "comparison")
    
    # After summarization, we can branch or just go sequentially. Let's do sequentially for simplicity
    workflow.add_edge("comparison", "citation")
    workflow.add_edge("citation", "research_gap")
    workflow.add_edge("research_gap", "compile_report")
    workflow.add_edge("compile_report", END)
    
    return workflow.compile()
