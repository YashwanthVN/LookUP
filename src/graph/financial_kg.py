import torch
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime
from torch_geometric.data import Data
from typing import List, Optional

class DynamicFinancialKG:
    # High-signal metrics for System 1
    METRIC_TYPES = ["revenue", "ebitda", "netProfitMargin", "eps", "pe_ratio"]

    def __init__(self, quarters: int = 4, api_key: Optional[str] = None):
        from src.streaming.financial_client import FMPClient # Local import to avoid circularity
        self.client = FMPClient(api_key)
        self.graph = nx.MultiDiGraph()
        self.quarters = quarters

    def build_for_tickers(self, tickers: List[str]):
        self.graph.clear()
        print(f"Step 1: Fetching bulk data for {len(tickers)} tickers...")
        bulk_data = self.client.get_bulk_data(tickers, self.quarters)
        profiles = bulk_data["profiles"]
        all_fin = bulk_data["financials"]
        
        all_quarters = set()

        # 1. Build Nodes
        print("Step 2: Building nodes...")
        for ticker in tickers:
            prof = profiles.get(ticker, {})
            
            # Sector is key for competitor edges
            self.graph.add_node(f"company_{ticker}", type="company", sector=prof.get("sector"), feat=[1.0, 0.0, 0.0])
            fin_df = all_fin.get(ticker, pd.DataFrame())
            if not fin_df.empty:
                all_quarters.update(fin_df['date'].astype(str).tolist())

        for m in self.METRIC_TYPES:
            self.graph.add_node(f"metric_{m}", type="metric", feat=[0.0, 1.0, 0.0])

        for q in sorted(all_quarters):
            self.graph.add_node(f"time_{q}", type="time", feat=[0.0, 0.0, 1.0])

        # 2. Build Edges (System 1: Hard Numbers)
        for ticker in tickers:
            fin_df = all_fin.get(ticker, pd.DataFrame())
            for _, row in fin_df.iterrows():
                d_str = str(row['date'])
                for m in self.METRIC_TYPES:
                    if m in row:
                        self._add_metric_edges(ticker, m, d_str, row[m])

        # 3. Add Competitor Edges
        print("Step 3: Building competitor edges...")
        self._add_competitor_edges()
        
        # 4. Normalize
        print("Step 4: Normalizing metrics...")
        self._normalize_metrics()

    def add_news_event(self, ticker, sentiment, magnitude):
        """ System 2: News Node Injection """
        news_id = f"news_{ticker}_{datetime.now().timestamp()}"
        self.graph.add_node(news_id, type="news", feat=[0.0, 0.0, 1.0], sentiment=sentiment)
        self.graph.add_edge(news_id, f"company_{ticker}", relation="IMPACTS", value=magnitude)

    def _add_competitor_edges(self):
        companies = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'company']
        for i, c1 in enumerate(companies):
            for c2 in companies[i+1:]:
                if self.graph.nodes[c1].get('sector') == self.graph.nodes[c2].get('sector'):
                    self.graph.add_edge(c1, c2, relation="COMPETES_WITH", value=1.0)

    def _normalize_metrics(self):
        # We only normalize numerical values on edges
        values = [d['value'] for _, _, d in self.graph.edges(data=True) if 'value' in d]
        if not values: return
        mean, std = np.mean(values), np.std(values)
        for _, _, d in self.graph.edges(data=True):
            if 'value' in d:
                d['value'] = (d['value'] - mean) / (std if std > 0 else 1.0)

    def _add_metric_edges(self, ticker, metric, date, val):
        comp_node, met_node, t_node = f"company_{ticker}", f"metric_{metric}", f"time_{date}"
        self.graph.add_edge(comp_node, met_node, relation="HAS_METRIC")
        self.graph.add_edge(met_node, t_node, relation="VALUE_AT", value=float(val or 0.0))
        
    @property
    def node_labels(self):
        """Returns a dictionary mapping index to node name for easy lookup"""
        return {i: name for i, name in enumerate(self.graph.nodes)}

    def get_node_index(self, label: str):
        """Helper to find the index of a specific company or metric"""
        node_list = list(self.graph.nodes)
        try:
            # Automatically handles the prefixing for you
            if label in self.METRIC_TYPES:
                return node_list.index(f"metric_{label}")
            return node_list.index(f"company_{label}")
        except ValueError:
            return None
    def add_sector_news(self, sector_name: str, sentiment: float, magnitude: float):
        """
        System 2: Multi-hop Injection. 
        Impacts a sector hub, which ripples to all companies in that sector.
        """
        news_id = f"sector_news_{sector_name}_{datetime.now().timestamp()}"
        self.graph.add_node(news_id, type="news", feat=[0.0, 0.0, 1.0], sentiment=sentiment)
    
        # ALL of the following lines must be indented inside the method!
        sector_companies = [n for n, d in self.graph.nodes(data=True) 
                            if d.get('type') == 'company' and d.get('sector') == sector_name]
        
        for comp in sector_companies:
            self.graph.add_edge(news_id, comp, relation="SECTOR_IMPACT", value=magnitude)
        
        print(f"Injected news for {sector_name} sector. Impacted {len(sector_companies)} companies.")

    def to_pyg_data(self) -> Data:
        node_list = list(self.graph.nodes)
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        x = torch.tensor([self.graph.nodes[n].get('feat', [0,0,0]) for n in node_list], dtype=torch.float)
        
        edge_index, edge_attr = [], []
        for u, v, d in self.graph.edges(data=True):
            edge_index.append([node_to_idx[u], node_to_idx[v]])
            edge_attr.append([d.get('value', 0.0)])
            
        return Data(x=x, edge_index=torch.tensor(edge_index).t().contiguous(), 
                    edge_attr=torch.tensor(edge_attr, dtype=torch.float))