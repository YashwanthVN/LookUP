import torch
import networkx as nx
import pandas as pd
from datetime import datetime
from torch_geometric.data import Data
from typing import List, Optional
from src.streaming.financial_client import FMPClient

class DynamicFinancialKG:
    METRIC_TYPES = ["pe_ratio", "revenue", "ebitda", "profit_margin", "market_cap", "eps"]

    def __init__(self, quarters: int = 4, api_key: Optional[str] = None):
        self.client = FMPClient(api_key)
        self.graph = nx.MultiDiGraph()
        self.quarters = quarters

    def build_for_tickers(self, tickers: List[str]):
        self.graph.clear()
        
        # ATOMIC FETCH: One call to rule them all
        bulk_data = self.client.get_bulk_data(tickers, self.quarters)
        profiles = bulk_data["profiles"]
        all_fin = bulk_data["financials"]
        
        all_quarters = set()

        # 1. Build Nodes
        for ticker in tickers:
            prof = profiles.get(ticker, {})
            self.graph.add_node(f"company_{ticker}", type="company", sector=prof.get("sector"), feat=[1.0, 0.0, 0.0])
            
            # Extract time points from this ticker's financial data
            fin_df = all_fin.get(ticker, pd.DataFrame())
            if not fin_df.empty:
                all_quarters.update(fin_df['date'].astype(str).tolist())

        for m in self.METRIC_TYPES:
            self.graph.add_node(f"metric_{m}", type="metric", feat=[0.0, 1.0, 0.0])

        if not all_quarters: all_quarters.add(datetime.now().strftime('%Y-%m-%d'))
        for q in sorted(all_quarters):
            self.graph.add_node(f"time_{q}", type="time", feat=[0.0, 0.0, 1.0])

        # 2. Build Edges from Local Data
        for ticker in tickers:
            prof = profiles.get(ticker, {})
            fin_df = all_fin.get(ticker, pd.DataFrame())

            # Quarterly Edges
            for _, row in fin_df.iterrows():
                d_str = str(row['date'])
                if 'eps' in row: self._add_metric_edges(ticker, "eps", d_str, row['eps'])
                if 'revenue' in row: self._add_metric_edges(ticker, "revenue", d_str, row['revenue'])

            # Profile/Static Edges
            for q_date in all_quarters:
                if prof.get("pe"): self._add_metric_edges(ticker, "pe_ratio", q_date, prof["pe"])
                if prof.get("mktCap"): self._add_metric_edges(ticker, "market_cap", q_date, prof["mktCap"])

    def _add_metric_edges(self, ticker, metric, date, val):
        comp_node = f"company_{ticker}"
        met_node = f"metric_{metric}"
        t_node = f"time_{date}"
        
        if comp_node not in self.graph:
            self.graph.add_node(comp_node, type="company", feat=[1.0, 0.0, 0.0])
        if met_node not in self.graph:
            self.graph.add_node(met_node, type="metric", feat=[0.0, 1.0, 0.0])
        if t_node not in self.graph:
            self.graph.add_node(t_node, type="time", feat=[0.0, 0.0, 1.0])
        
        self.graph.add_edge(comp_node, met_node, relation="HAS_METRIC")
        self.graph.add_edge(met_node, t_node, relation="VALUE_AT", value=float(val or 0.0))

    def to_pyg_data(self) -> Data:
        node_list = list(self.graph.nodes)
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        
        # Use .get() with a default [0.0, 0.0, 0.0] to avoid KeyError
        x_list = []
        for n in node_list:
            feat = self.graph.nodes[n].get('feat', [0.0, 0.0, 0.0])
            x_list.append(feat)
        
        x = torch.tensor(x_list, dtype=torch.float)

        edge_index = []
        edge_attr = []
        for u, v, attrs in self.graph.edges(data=True):
            edge_index.append([node_to_idx[u], node_to_idx[v]])
            edge_attr.append([attrs.get('value', 0.0)])

        if not edge_index:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)