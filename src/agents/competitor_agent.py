import torch
import torch.nn.functional as F
from .state import LOOKUPState
from src.kg_holder import get_kg

def competitor_agent(state: LOOKUPState) -> dict:
    kg = get_kg()
    if kg is None: return {"competitor_candidates": []}

    primary_ticker = state["tickers"][0]
    
    # 1. Sync Data and Node List
    data = kg.to_pyg_data()
    node_list = list(kg.graph.nodes)
    
    # DEBUG: Let's see what the agent actually sees
    company_nodes = [n for n in node_list if "company_" in n]
    print(f"🔍 AGENT DEBUG: Found {len(company_nodes)} total company nodes in graph.")

    # 2. Forward Pass for Embeddings
    kg.gnn_model.eval()
    with torch.no_grad():
        # returns z, (edge_index, alpha)
        embeddings, _ = kg.gnn_model(data.x, data.edge_index, data.edge_attr, return_attention=True)
    
    # 3. Robust Node Lookup
    target_idx = kg.get_node_index(primary_ticker)
    
    if target_idx is None:
        print(f"⚠️ DEBUG: Target {primary_ticker} not found. Available: {company_nodes[:3]}")
        return {"competitor_candidates": []}

    primary_emb = embeddings[target_idx].unsqueeze(0)
    
    # 4. Find ALL company nodes and compute similarity
    similarities = []
    for i, node_name in enumerate(node_list):
        # Match anything containing 'company_' that isn't our primary
        if "company_" in node_name and i != target_idx:
            other_emb = embeddings[i].unsqueeze(0)
            sim = F.cosine_similarity(primary_emb, other_emb).item()
            ticker_label = node_name.replace("company_", "")
            similarities.append((ticker_label, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    candidates = [f"{t} (Sim: {s:.4f})" for t, s in similarities[:5]]
    
    print(f"✅ Competitor Agent found {len(candidates)} candidates.")
    return {"competitor_candidates": candidates}