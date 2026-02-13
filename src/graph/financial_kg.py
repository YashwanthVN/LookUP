import torch
from torch_geometric.data import Data
import networkx as nx

class FinancialKG:
    """Build and query financial knowledge graph"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        
    def build_mini_kg(self):
        """3-company, 2-year prototype KG"""
        # ... implementation from your earlier notes
        pass
    
    def to_pyg_data(self) -> Data:
        """Convert current graph to PyG Data object"""
        # ... conversion logic
        pass