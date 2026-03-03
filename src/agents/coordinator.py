from typing import TypedDict, Annotated, List, Literal
import operator
import torch
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import agent functions
from .gnn_agent import gnn_rag_agent
from .competitor_agent import competitor_agent
from .temporal_analyst import temporal_analyst
from .retriever_agent import retriever_agent
from .writer import writer_agent
from .state import LOOKUPState

# Import kg holder to access global kg
from src.kg_holder import get_kg

def coordinator_agent(state: LOOKUPState) -> dict:
    kg = get_kg()
    query_lower = state["query"].lower()
    tickers = state["tickers"] # e.g., ["AAPL"]

    # 1. PEER DISCOVERY: If it's a competitor query, add peers to the graph
    if "competitor" in query_lower or "rival" in query_lower:
        primary = tickers[0]
        # In a real scenario, you'd use: peers = kg.client.get_peers(primary)
        # For this week, let's ensure these 5 are always in the comparison set
        peers = ["MSFT", "GOOGL", "NVDA", "AMZN"]
        tickers = list(set(tickers + peers))
    
    # 2. Build KG for the expanded list
    # This prevents the graph from having only 1 company node
    kg.build_for_tickers(tickers)
    
    for t in tickers:
        kg.inject_real_time_news(t)

    # 3. Standard Routing
    if "competitor" in query_lower or "rival" in query_lower:
        return {"next_step": "competitor", "tickers": tickers}
    
    if "news" in query_lower or "article" in query_lower or "information" in query_lower:
        return {"next_step": "retriever"}
    return {"next_step": "gnn"}

def route_next_agent(state: LOOKUPState) -> Literal["gnn", "temporal", "competitor", "write", "retriever"]:
    # This must match the key returned by coordinator_agent
    return state.get("next_step", "write")

def create_graph():
    workflow = StateGraph(LOOKUPState)

    workflow.add_node("coordinator", coordinator_agent)
    workflow.add_node("gnn", gnn_rag_agent)
    workflow.add_node("competitor", competitor_agent)   # <-- new node
    workflow.add_node("temporal", temporal_analyst)     # (stub)
    workflow.add_node("retriever", retriever_agent)
    workflow.add_node("write", writer_agent)

    workflow.set_entry_point("coordinator")

    workflow.add_conditional_edges(
        "coordinator",
        route_next_agent,
        {
            "gnn": "gnn",
            "competitor": "competitor",
            "temporal": "temporal",
            "retriever": "retriever",
            "write": "write"
        }
    )

    workflow.add_edge("gnn", "write")
    workflow.add_edge("competitor", "write")
    workflow.add_edge("temporal", "write")
    workflow.add_edge("retriever", "write")
    workflow.add_edge("write", END)
    return workflow.compile(checkpointer=MemorySaver())