from typing import TypedDict, Annotated, List, Literal
import operator
import torch
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import agent functions
from .gnn_agent import gnn_rag_agent
from .competitor_agent import competitor_agent
from .temporal_analyst import temporal_analyst
from .writer import writer_agent
from .state import LOOKUPState

# Import kg holder to access global kg
from src.kg_holder import get_kg

def coordinator_agent(state: LOOKUPState) -> dict:
    kg = get_kg()
    if kg is None: return {"final_report": "Error: KG not found"}
    
    # 1. Build and Inject
    kg.build_for_tickers(state["tickers"])
    for ticker in state["tickers"]:
        kg.inject_real_time_news(ticker)
    
    # 2. MATCHING LOGIC: Use 'next_step' as the decision key
    query_lower = state["query"].lower()
    if any(x in query_lower for x in ["compare", "versus", "vs"]):
        return {"next_step": "gnn"}
    elif "trend" in query_lower:
        return {"next_step": "temporal"}
    else:
        return {"next_step": "gnn"}

def route_next_agent(state: LOOKUPState) -> Literal["gnn", "temporal", "competitor", "write"]:
    # This must match the key returned by coordinator_agent
    return state.get("next_step", "write")

def create_graph():
    workflow = StateGraph(LOOKUPState)
    workflow.add_node("coordinator", coordinator_agent)
    workflow.add_node("gnn", gnn_rag_agent)
    workflow.add_node("write", writer_agent)
    
    workflow.set_entry_point("coordinator")
    
    # The keys in this dictionary MUST match the strings returned by route_next_agent
    workflow.add_conditional_edges(
        "coordinator",
        route_next_agent,
        {
            "gnn": "gnn",
            "write": "write"
        }
    )
    workflow.add_edge("gnn", "write")
    workflow.add_edge("write", END)
    return workflow.compile(checkpointer=MemorySaver())