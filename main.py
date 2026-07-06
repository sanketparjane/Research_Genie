import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from .graph import create_research_graph
import asyncio

app = FastAPI(title="ResearchGenie API")
graph = create_research_graph()

class QueryRequest(BaseModel):
    query: str

async def generate_workflow_stream(query: str):
    initial_state = {"query": query}
    
    # Run the graph and yield steps
    for output in graph.stream(initial_state):
        # output is a dict like {'planner': {'subtasks': [...], 'current_step': '...'}}
        for node_name, state_update in output.items():
            step_info = state_update.get("current_step", f"{node_name} finished.")
            yield f"data: {json.dumps({'step': step_info, 'node': node_name})}\n\n"
            
            # If compile_report, yield the final report
            if node_name == "compile_report":
                final_report = state_update.get("final_report", "")
                yield f"data: {json.dumps({'final_report': final_report})}\n\n"
        await asyncio.sleep(0.1)

@app.post("/research")
async def run_research(request: QueryRequest):
    return StreamingResponse(
        generate_workflow_stream(request.query),
        media_type="text/event-stream"
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}
