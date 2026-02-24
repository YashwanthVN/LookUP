import torch
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime
from torch_geometric.data import Data
from typing import List, Optional

class DynamicFinancialKG:
    METRIC_TYPES = ["revenue", "ebitda", "netProfitMargin", "eps", "pe_ratio"]

    def __init__(self, quarters: int = 4, api_key: Optional[str] = None):
        from src.streaming.financial_client import FMPClient
        self.client = FMPClient(api_key)
        self.graph = nx.MultiDiGraph()
        self.quarters = quarters

    def build_for_tickers(self, tickers: List[str]):
        self.graph.clear()
        print(f"Step 1: Fetching bulk data for {len(tickers)} tickers...")
        bulk_data = self.client.get_bulk_data(tickers, self.quarters)
        profiles = bulk_data["profiles"]
        all_fin = bulk_data["financials"]
        
        # 1. Build Nodes
        print("Step 2: Building nodes...")
        for ticker in tickers:
            prof = profiles.get(ticker, {})
            self.graph.add_node(f"company_{ticker}", 
                               type="company", 
                               sector=prof.get("sector"), 
                               feat=[1.0, 0.0, 0.0])

        for m in self.METRIC_TYPES:
            self.graph.add_node(f"metric_{m}", type="metric", feat=[0.0, 1.0, 0.0])

        # 2. Build System 1 Edges (Hard Numbers)
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
        
        # 4. Normalize (Per-Metric Deep Fix)
        print("Step 4: Normalizing metrics...")
        self._normalize_metrics()

    def _add_metric_edges(self, ticker, metric, date, val):
        """FIX: Direct company-to-metric value for System 1 visibility."""
        comp_node, met_node = f"company_{ticker}", f"metric_{metric}"
        self.graph.add_edge(
            comp_node, met_node, 
            relation="REPORTED", 
            value=float(val or 0.0),
            date=date,
            is_metric=True
        )

    def _add_competitor_edges(self):
        """Optimization: Bi-directional sector links."""
        companies = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'company']
        for i, c1 in enumerate(companies):
            for c2 in companies[i+1:]:
                if self.graph.nodes[c1].get('sector') == self.graph.nodes[c2].get('sector'):
                    self.graph.add_edge(c1, c2, relation="COMPETES_WITH", value=0.5)
                    self.graph.add_edge(c2, c1, relation="COMPETES_WITH", value=0.5)

    def _normalize_metrics(self):
        """Deep Fix: Z-Score normalization per metric category."""
        for metric_type in self.METRIC_TYPES:
            edges = []
            for u, v, k, d in self.graph.edges(keys=True, data=True):
                if f"metric_{metric_type}" in v and d.get('is_metric'):
                    edges.append(d)
            
            vals = [e['value'] for e in edges]
            if not vals: continue
            mean, std = np.mean(vals), np.std(vals)
            for e in edges:
                e['value'] = (e['value'] - mean) / (std if std > 0 else 1.0)

    def add_news_event(self, ticker, sentiment, magnitude):
        news_id = f"news_{ticker}_{datetime.now().timestamp()}"
        self.graph.add_node(news_id, type="news", feat=[0.0, 0.0, 1.0], sentiment=sentiment)
        self.graph.add_edge(news_id, f"company_{ticker}", relation="IMPACTS", value=magnitude)

    def inject_real_time_news(self, ticker: str):
        news_data = self.client.get_stock_news(ticker)
        bullish = ['beat', 'growth', 'surge', 'buy', 'positive']
        bearish = ['miss', 'fall', 'drop', 'sell', 'negative']
        
        total_sentiment = 0
        for article in news_data:
            text = article.get('text', '').lower()
            total_sentiment += sum([1 for w in bullish if w in text])
            total_sentiment -= sum([1 for w in bearish if w in text])

        consensus = np.clip(total_sentiment / len(news_data), -1, 1) if news_data else 0
        self.add_news_event(ticker, sentiment=consensus, magnitude=abs(total_sentiment))

    @property
    def node_labels(self):
        return {i: name for i, name in enumerate(self.graph.nodes)}

    def get_node_index(self, label: str):
        node_list = list(self.graph.nodes)
        try:
            if label in self.METRIC_TYPES: return node_list.index(f"metric_{label}")
            if not label.startswith("company_"): label = f"company_{label}"
            return node_list.index(label)
        except ValueError: return None

    def to_pyg_data(self, decay_lambda=0.5) -> Data:
        node_list = list(self.graph.nodes)
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        current_time = datetime.now().timestamp()
        
        x = torch.tensor([self.graph.nodes[n].get('feat', [0,0,0]) for n in node_list], dtype=torch.float)
        
        edge_index, edge_attr = [], []
        for u, v, d in self.graph.edges(data=True):
            val = d.get('value', 1.0)
            
            # FIX: Check for 'created_at' (used in your tests) OR 'timestamp'
            t_key = 'created_at' if 'created_at' in d else 'timestamp'
            
            if t_key in d:
                delta_t = (current_time - d[t_key]) / 3600 
                val = val * np.exp(-decay_lambda * delta_t)
                if val < 0.01: continue 

            edge_index.append([node_to_idx[u], node_to_idx[v]])
            edge_attr.append([float(val)])
            
        return Data(x=x, 
                    edge_index=torch.tensor(edge_index).t().contiguous(), 
                    edge_attr=torch.tensor(edge_attr, dtype=torch.float))