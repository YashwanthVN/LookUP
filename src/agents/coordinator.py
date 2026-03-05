from typing import TypedDict, Annotated, List, Literal
import operator
import torch
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import yfinance as yf

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
    
    # 1. Clear for fresh session
    kg.graph.clear()
    try:
        existing_data = kg.vector_store.collection.get()
        if existing_data['ids']:
            kg.vector_store.collection.delete(ids=existing_data['ids'])
    except Exception as e:
        print(f"⚠️ Vector store clear skipped: {e}")
    
    query_lower = state["query"].lower()
    tickers = state["tickers"]
    primary = tickers[0]
    
    # 2. Query Expansion
    expanded_query = f"{primary} stock analysis financial news March 2026"
    
    # 3. BUILD AND INJECT (Crucial: Do this BEFORE returning)
    # This ensures the data is ready for BOTH GNN and Retriever
    kg.build_for_tickers(tickers)
    for t in tickers:
        kg.inject_real_time_news(t)

    # 4. Return the updated state
    return {
        "query": expanded_query, 
        "tickers": tickers,
        "next_step": "retriever" # Start the chain
    }

def route_next_agent(state: LOOKUPState) -> Literal["gnn", "temporal", "competitor", "write", "retriever"]:
    # This must match the key returned by coordinator_agent
    return state.get("next_step", "write")

def create_graph():
    workflow = StateGraph(LOOKUPState)

    workflow.add_node("coordinator", coordinator_agent)
    workflow.add_node("retriever", retriever_agent)
    workflow.add_node("gnn", gnn_rag_agent)
    workflow.add_node("write", writer_agent)

    workflow.set_entry_point("coordinator")

    # Define a LINEAR CHAIN so data accumulates in the 'state'
    workflow.add_edge("coordinator", "retriever")
    workflow.add_edge("retriever", "gnn")
    workflow.add_edge("gnn", "write")
    workflow.add_edge("write", END)

    return workflow.compile(checkpointer=MemorySaver())