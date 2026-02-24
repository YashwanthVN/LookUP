import torch
import torch.nn as nn
from torch_geometric.nn import NNConv, GATv2Conv

class FinancialReasoningGNN(nn.Module):
    def __init__(self, in_channels=3, edge_channels=1, hidden=64, out=32):
        super().__init__()
        
        # --- System 1: Structural (NNConv) ---
        # Maps edge metrics (Revenue, EPS) to a weight matrix for node transformation
        edge_mlp = nn.Sequential(
            nn.Linear(edge_channels, hidden * in_channels),
            nn.ReLU(),
            nn.Linear(hidden * in_channels, in_channels * hidden)
        )
        self.sys1_structural = NNConv(in_channels, hidden, edge_mlp)
        
        # --- System 2: Narrative (GATv2) ---
        # Uses attention to focus on "noisy" news impacts
        self.sys2_narrative = GATv2Conv(in_channels, hidden, heads=4, edge_dim=edge_channels)
        
        # --- The Merger (MLP) ---
        # Concatenates [Structural Embedding + Narrative Embedding]
        # hidden (Sys1) + (hidden * 4 heads for Sys2) = hidden * 5
        self.merger = nn.Sequential(
            nn.Linear(hidden + (hidden * 4), hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, out)
        )
        
    def forward(self, x, edge_index, edge_attr, return_attention=False):
        # 1. Run Systems in Parallel
        # System 1: Focuses on the "magnitude" of financial edges
        h_structural = self.sys1_structural(x, edge_index, edge_attr).relu()
        
        # System 2: Focuses on the "attention" of news/sector edges
        h_narrative, (edge_index_attn, alpha) = self.sys2_narrative(
            x, edge_index, edge_attr, return_attention_weights=True
        )
        h_narrative = h_narrative.relu()
        
        # 2. Concatenate (Fusing the two "brains")
        combined = torch.cat([h_structural, h_narrative], dim=-1)
        
        # 3. Final MLP Projection
        z = self.merger(combined)
        
        if return_attention:
            return z, (edge_index_attn, alpha)
        return z