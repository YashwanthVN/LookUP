from typing import TypedDict, Annotated, List, Literal
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver

class LOOKUPState(TypedDict):
    """Shared state across all agents"""
    query: str
    tickers: List[str]
    competitor_candidates: List[str]
    gnn_evidence: Annotated[List[str], operator.add]
    temporal_insights: str
    reranked_documents: List[str]
    final_report: str
    iteration_count: int

def coordinator_agent(state: LOOKUPState) -> dict:
    """Route to appropriate specialist based on query analysis"""
    # Simple stub – will be expanded
    if "compare" in state["query"].lower():
        return {"next": "gnn"}
    return {"next": "write"}

def route_next_agent(state: LOOKUPState) -> Literal["gnn", "temporal", "competitor", "write"]:
    """Conditional edge from coordinator"""
    return state.get("next", "write")

def create_graph():
    """Build the multi-agent workflow"""
    workflow = StateGraph(LOOKUPState)
    
    # Add nodes (will be imported and added later)
    workflow.add_node("coordinator", coordinator_agent)
    # ... other nodes added via separate functions
    
    return workflow.compile(checkpointer=MemorySaver())