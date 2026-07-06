from typing import List, Dict, Any, TypedDict, Annotated
import operator

class ResearchState(TypedDict):
    query: str
    subtasks: List[str]
    search_results: List[Dict[str, Any]]
    downloaded_pdfs: List[str]
    extracted_texts: Dict[str, str]
    rag_context: str
    summaries: Dict[str, str]
    comparison: str
    citations: str
    research_gaps: str
    final_report: str
    current_step: str
