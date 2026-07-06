import os
import urllib.request
import fitz  # PyMuPDF
import arxiv
from typing import List, Dict, Any

def search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search ArXiv for papers matching the query.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    for r in client.results(search):
        results.append({
            "title": r.title,
            "authors": [author.name for author in r.authors],
            "summary": r.summary,
            "published": r.published.strftime("%Y-%m-%d"),
            "pdf_url": r.pdf_url,
            "entry_id": r.entry_id
        })
    return results

def download_pdf(pdf_url: str, save_dir: str = "./data/pdfs") -> str:
    """
    Download a PDF from a URL and save it locally.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = pdf_url.split('/')[-1]
    if not filename.endswith('.pdf'):
        filename += '.pdf'
        
    filepath = os.path.join(save_dir, filename)
    
    if not os.path.exists(filepath):
        urllib.request.urlretrieve(pdf_url, filepath)
        
    return filepath

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file using PyMuPDF.
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text
