import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class FinancialGNNRAG(torch.nn.Module):
    """GNN for dense subgraph retrieval (Mavromatis & Karypis, 2024)"""
    
    def __init__(self, in_channels=3, hidden=64, out=32):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden, heads=4)
        self.conv2 = GATConv(hidden*4, hidden, heads=4)
        self.conv3 = GATConv(hidden*4, out, heads=1, concat=False)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)
        return x
    
    # ... scoring & path extraction methods (add later)