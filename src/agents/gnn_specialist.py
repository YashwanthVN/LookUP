from .coordinator import LOOKUPState

def gnn_rag_agent(state: LOOKUPState) -> dict:
    """Execute GNN‑RAG dense subgraph reasoning"""
    # Stub: will call src.graph.gnn_rag
    print("GNN specialist working...")
    return {
        "gnn_evidence": ["AAPL HAS_METRIC pe_ratio VALUE 29.5 → ..."],
        "competitor_candidates": ["MSFT", "GOOGL"]
    }