import torch
import torch.nn as nn
from torch_geometric.nn import NNConv, GATv2Conv

class FinancialReasoningGNN(nn.Module):
    def __init__(self, in_channels=3, edge_channels=1, hidden=64, out=32):
        super().__init__()
        
        # System 1: Magnitude Reasoning
        # The MLP takes the edge value (e.g., Z-score of Revenue) and creates a weight matrix
        edge_mlp = nn.Sequential(
            nn.Linear(edge_channels, hidden * in_channels),
            nn.ReLU(),
            nn.Linear(hidden * in_channels, in_channels * hidden)
        )
        self.conv1 = NNConv(in_channels, hidden, edge_mlp)
        
        # System 2: Contextual Attention
        # GATv2 focuses on "important" connections (like News impacts)
        self.conv2 = GATv2Conv(hidden, hidden, heads=4, edge_dim=edge_channels)
        self.conv3 = GATv2Conv(hidden * 4, out, heads=1, concat=False)
        
    def forward(self, x, edge_index, edge_attr, return_attention=False):
        # Step 1: System 1 (Magnitude Reasoning)
        x = self.conv1(x, edge_index, edge_attr).relu()
        
        # Step 2: System 2 (Contextual Attention) - This is where we get weights
        # We get attention from conv2 because it has the multi-head context
        x, (edge_index_attn, alpha) = self.conv2(x, edge_index, edge_attr, return_attention_weights=True)
        x = x.relu()
        
        # Step 3: Final Projection
        x = self.conv3(x, edge_index)
        
        if return_attention:
            return x, (edge_index_attn, alpha)
        return x