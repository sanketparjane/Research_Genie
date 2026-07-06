import streamlit as st
import requests
import json
from fpdf import FPDF

st.set_page_config(page_title="ResearchGenie", page_icon="🧞‍♂️", layout="wide")

st.title("🧞‍♂️ ResearchGenie: Multi-Agent AI Research Assistant")
st.markdown("Enter a research topic, and our team of AI agents will find ArXiv papers, analyze them, and generate a comprehensive literature review.")

query = st.text_input("Enter your research topic:", placeholder="e.g., Multi-Agent Reinforcement Learning in Robotics")

if st.button("Start Research") and query:
    st.divider()
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    report_container = st.empty()
    final_report = ""
    
    # Define agent steps for progress tracking
    steps = [
        "planner", "search", "pdf", "rag", 
        "summarization", "comparison", "citation", 
        "research_gap", "compile_report"
    ]
    
    try:
        with requests.post("http://localhost:8000/research", json={"query": query}, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        data = json.loads(decoded_line[6:])
                        
                        if "step" in data:
                            node = data.get("node")
                            step_msg = data.get("step")
                            
                            status_text.info(f"⏳ **{node.upper()}**: {step_msg}")
                            
                            if node in steps:
                                progress_bar.progress((steps.index(node) + 1) / len(steps))
                                
                        if "final_report" in data:
                            final_report = data["final_report"]
                            status_text.success("✨ Research Completed!")
                            report_container.markdown(final_report)
                            
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")

    if final_report:
        # Create PDF for download
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # very basic pdf generation (handling ascii only for simplicity)
        encoded_report = final_report.encode('latin-1', 'replace').decode('latin-1')
        for line in encoded_report.split('\n'):
            pdf.multi_cell(0, 10, txt=line)
            
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        
        st.download_button(
            label="📄 Download Literature Review (PDF)",
            data=pdf_bytes,
            file_name="literature_review.pdf",
            mime="application/pdf"
        )
