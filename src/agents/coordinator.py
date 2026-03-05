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
    
    kg.graph.clear()
    try:
        # Get all existing IDs in the collection
        existing_data = kg.vector_store.collection.get()
        if existing_data['ids']:
            kg.vector_store.collection.delete(ids=existing_data['ids'])
    except Exception as e:
        print(f"⚠️ Vector store clear skipped or failed: {e}")
    
    query_lower = state["query"].lower()
    tickers = state["tickers"]
    primary = state["tickers"][0]
    
    if "news" in query_lower or "how is" in query_lower:
        # Transform "how is Ola" -> "Ola Electric Mobility stock performance news March 2026"
        expanded_query = f"{primary} stock analysis financial news March 2026"
        state["query"] = expanded_query
        return {"next_step": "retriever", "query": expanded_query}

    if "competitor" in query_lower or "rival" in query_lower:
        try:
            # Dynamic Peer Discovery
            ticker_obj = yf.Ticker(primary)
            sector = ticker_obj.info.get('sector', '')
            
            # Fallback Peer Lists per Sector
            if "Aerospace" in sector or primary == "BA":
                peers = ["LMT", "RTX", "NOC", "GD"]
            elif "Technology" in sector:
                peers = ["MSFT", "GOOGL", "NVDA", "AAPL"]
            else:
                # If sector unknown, use basic market leaders as anchors
                peers = ["SPY", "QQQ"] 
                
            tickers = list(set(tickers + peers))
        except:
            pass 

    # QUERY EXPANSION: Broden the search for the retriever
    if "retriever" in state.get("next_step", "") or "how is" in query_lower:
        state["query"] = f"{primary} stock analysis performance news 2026"

    # Build KG and Inject
    kg.build_for_tickers(tickers)
    for t in tickers:
        kg.inject_real_time_news(t)

    if "competitor" in query_lower or "rival" in query_lower:
        return {"next_step": "competitor", "tickers": tickers}
    
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