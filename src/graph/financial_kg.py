import torch
from torch_geometric.data import Data
import networkx as nx

class FinancialKG:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        
    def build_mini_kg(self):
        """3 companies, 2 metrics, 2 time periods – hardcoded for testing"""
        # Nodes: companies, metrics, time
        companies = ["AAPL", "MSFT", "GOOGL"]
        metrics = ["pe_ratio", "revenue"]
        quarters = ["2024Q1", "2024Q2"]
        
        # Add nodes with features (random for now)
        node_list = []
        for c in companies:
            self.graph.add_node(f"company_{c}", type="company", feat=[1.0, 0.0, 0.0])
        for m in metrics:
            self.graph.add_node(f"metric_{m}", type="metric", feat=[0.0, 1.0, 0.0])
        for q in quarters:
            self.graph.add_node(f"time_{q}", type="time", feat=[0.0, 0.0, 1.0])
            
        # Edges: AAPL → pe_ratio (Q2) → 29.5
        self.graph.add_edge("company_AAPL", "metric_pe_ratio", relation="HAS_METRIC")
        self.graph.add_edge("metric_pe_ratio", "time_2024Q2", relation="MEASURED_AT", value=29.5)
        # ... add more edges
        
    def to_pyg_data(self):
        """Convert current graph to PyG Data object"""
        # Map nodes to indices
        node_list = list(self.graph.nodes)
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        
        # Features
        x = torch.tensor([self.graph.nodes[n].get('feat', [0.0,0.0,0.0]) 
                          for n in node_list], dtype=torch.float)
        
        # Edge index
        edge_index = []
        for u, v in self.graph.edges:
            edge_index.append([node_to_idx[u], node_to_idx[v]])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)