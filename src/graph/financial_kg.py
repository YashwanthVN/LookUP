import os
import torch
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime
from torch_geometric.data import Data
from typing import List, Optional
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel
from pathlib import Path

import yfinance as yf
import requests
from datetime import datetime, timedelta
from src.graph.gnn_rag import FinancialReasoningGNN

class UnifiedNewsFetcher:
    def __init__(self, finnhub_api_key=None):
        self.finnhub_key = finnhub_api_key or os.getenv("FINNHUB_API_KEY")
        self.finnhub_url = "https://finnhub.io/api/v1/company-news"

    def fetch(self, ticker, limit=5):
        # 1. Try Finnhub first if key exists
        if self.finnhub_key:
            news = self._fetch_finnhub(ticker, limit)
            if news: return news

        # 2. Fallback to yfinance
        return self._fetch_yfinance(ticker, limit)

    def _fetch_finnhub(self, ticker, limit):
        try:
            end = datetime.now().strftime('%Y-%m-%d')
            start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            params = {'symbol': ticker, 'from': start, 'to': end, 'token': self.finnhub_key}
            r = requests.get(self.finnhub_url, params=params, timeout=10)
            if r.status_code == 200:
                return [{'headline': n.get('headline'), 'source': 'Finnhub'} for n in r.json()[:limit]]
        except: return None

    def _fetch_yfinance(self, ticker, limit):
        try:
            yf_ticker = yf.Ticker(ticker)
            return [{'headline': n.get('title'), 'source': 'yfinance'} for n in yf_ticker.news[:limit]]
        except: return []

class DynamicFinancialKG:
    METRIC_TYPES = ["revenue", "ebitda", "netProfitMargin", "eps", "pe_ratio"]

    def __init__(self, quarters: int = 4, api_key: Optional[str] = None):
        from src.streaming.financial_client import FMPClient
        self.client = FMPClient(api_key)
        self.graph = nx.MultiDiGraph()
        self.quarters = quarters
        # Load FinBERT once during initialization
        base_model_name = "yiyanghkust/finbert-tone"
        self.tokenizer = BertTokenizer.from_pretrained(base_model_name)
        base_model = BertForSequenceClassification.from_pretrained(base_model_name)
        
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent 
        adapter_path = project_root / "notebooks" / "finbert_sentiment_lora_final" 
        try:
            self.sentiment_model = PeftModel.from_pretrained(base_model, adapter_path)
            print("✅ Success: Loaded Fine-Tuned LoRA Sentiment Model.")
        except:
            print("⚠️ Warning: Could not find LoRA adapters. Falling back to vanilla FinBERT.")
            self.sentiment_model = base_model
            
        self.gnn_model = FinancialReasoningGNN(
            in_channels=3, 
            edge_channels=1, 
            hidden=64, 
            out=32
        )
        print("🧠 Reasoning GNN initialized.")

    def build_for_tickers(self, tickers: List[str]):
        self.tickers = tickers
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
        clean_val = float(val) if val is not None and np.isfinite(val) else 0.0
        
        self.graph.add_edge(
        comp_node, met_node, 
        relation="REPORTED", 
        value=clean_val,
        date=date,
        is_metric=True
    )

    def _add_competitor_edges(self):
        old_edges = [(u, v) for u, v, d in self.graph.edges(data=True) 
                 if d.get('relation') == "COMPETITOR_INFLUENCE"]
        self.graph.remove_edges_from(old_edges)
        """Adds edges between companies in the same sector weighted by Market Cap dominance."""
        for ticker in self.tickers:
            u_node = f"company_{ticker}"
            u_data = self.graph.nodes[u_node]
            u_mkt_cap = u_data.get('market_cap', 1.0) # Fallback to 1.0 if missing
            u_sector = u_data.get('sector')
            
            # Find all other nodes in the same sector
            sector_peers = [n for n, d in self.graph.nodes(data=True) 
                            if d.get('sector') == u_sector and n != u_node]
            
            if not sector_peers: continue
            
            # Calculate Sector Average Market Cap for normalization
            peer_caps = [self.graph.nodes[p].get('market_cap', 1.0) for p in sector_peers]
            avg_sector_cap = sum(peer_caps) / len(peer_caps)
            
            for peer_node in sector_peers:
                peer_mkt_cap = self.graph.nodes[peer_node].get('market_cap', 1.0)
                
                # RELATIVE DOMINANCE: How much bigger/smaller is this peer than the sector average?
                # We use a log-scale or a ratio to prevent trillion-dollar companies from breaking the GNN
                dominance_ratio = peer_mkt_cap / avg_sector_cap
                corr_weight = self._get_correlation_weight(ticker, peer_node.replace("company_", ""))
                final_value = dominance_ratio * corr_weight
                
                # The edge represents the INFLUENCE of the peer ON the target ticker
                self.graph.add_edge(
                peer_node, 
                u_node, 
                relation="COMPETITOR_INFLUENCE", 
                value=float(final_value),
                type="structural"
            )

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

    def add_news_event(self, ticker, sentiment, magnitude, headline=None):
        # 1. Generate unique ID
        news_id = f"news_{ticker}_{hash(headline) % 10000}" if headline else f"news_{ticker}_{datetime.now().timestamp()}"
        
        # 2. Add node with the keys the Engine actually uses: 'label' and 'sentiment'
        self.graph.add_node(
            news_id, 
            type="news", 
            feat=[0.0, 0.0, 1.0], 
            sentiment=float(sentiment or 0.0), # Store sentiment!
            label=str(headline or "No Headline"), # Store as 'label'
            source="Finnhub"
        )
        
        # 3. Connect to company
        self.graph.add_edge(news_id, f"company_{ticker}", relation="IMPACTS", value=magnitude)

    def inject_real_time_news(self, ticker: str):
        """Unified Fetcher + Advanced Graph Injection with BATCH Sentiment Inference"""
        print(f"📡 System 2: Scanning news for {ticker}...")
        
        # 1. Fetch news
        fetcher = UnifiedNewsFetcher(finnhub_api_key=os.getenv("FINNHUB_API_KEY"))
        news_items = fetcher.fetch(ticker)

        if not news_items:
            print(f"⚠️ No news found for {ticker}.")
            return

        # 2. Batch Sentiment Calculation
        headlines = [item['headline'] for item in news_items]
        inputs = self.tokenizer(headlines, return_tensors="pt", truncation=True, 
                                padding=True, max_length=512)
        
        device = next(self.sentiment_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        print(f"🧠 Running batch sentiment for {len(headlines)} headlines...")
        self.sentiment_model.eval()
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
        
        all_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # 3. Graph Injection via Helper Function
        for idx, item in enumerate(news_items):
            probs = all_probs[idx]
            # Calculate score (-1 to 1)
            score = probs[1].item() - probs[2].item()
            
            # Neutrality check
            if probs[0].item() > max(probs[1].item(), probs[2].item()) and abs(score) < 0.1:
                score = 0.0
            
            # Magnitude (Weight) calculation
            magnitude = abs(score) * 2.0 + 0.5
            
            # 🔥 THE FIX: Use the helper function so keys match the Reasoning Engine
            self.add_news_event(
                ticker=ticker,
                sentiment=score,
                magnitude=magnitude,
                headline=item['headline']
            )
            
        print(f"✅ System 2: Successfully injected {len(news_items)} news nodes for {ticker}.")

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

    def _get_correlation_weight(self, ticker_a, ticker_b):
        """Calculates Pearson Correlation between two tickers over last 30 days."""
        try:
            # Fetch historical prices (Simplified: assume client has get_prices)
            prices_a = self.client.get_historical_prices(ticker_a, limit=30)
            prices_b = self.client.get_historical_prices(ticker_b, limit=30)
            
            if len(prices_a) < 2 or len(prices_b) < 2: return 1.0
            
            # Calculate returns
            returns_a = np.diff(prices_a) / prices_a[:-1]
            returns_b = np.diff(prices_b) / prices_b[:-1]
            
            # Pearson Correlation
            corr = np.corrcoef(returns_a, returns_b)[0, 1]
            return max(0.1, corr) # Floor at 0.1 to keep the edge alive
        except:
            return 0.5 # Default if API fails
    
    def to_pyg_data(self, decay_lambda=0.5) -> Data:
        node_list = list(self.graph.nodes)
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        current_time = datetime.now().timestamp()
        
        x = torch.tensor([self.graph.nodes[n].get('feat', [0,0,0]) for n in node_list], dtype=torch.float)
        
        edge_index, edge_attr = [], []
        for u, v, d in self.graph.edges(data=True):
            val = d.get('value', 1.0)
            
            # Ensure val is a valid number before decay
            if not np.isfinite(val): val = 0.0
            
            # ... (time decay logic) ...

            edge_index.append([node_to_idx[u], node_to_idx[v]])
            edge_attr.append([float(val)])
        
        # Final check: Convert any missed nans in the tensor to 0
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float)
        edge_attr_t = torch.nan_to_num(edge_attr_t, nan=0.0)
        
        return Data(x=x, 
                    edge_index=torch.tensor(edge_index).t().contiguous(), 
                    edge_attr=edge_attr_t)
        
    def _analyze_sentiment(self, text: str):
        """
        Fixed mapping for yiyanghkust/finbert-tone:
        0: Neutral, 1: Positive, 2: Negative
        """
        # Truncate and move to same device as model
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        device = next(self.sentiment_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
        
        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        
        # Index 1 = Positive, Index 2 = Negative
        pos_score = probs[1].item()
        neg_score = probs[2].item()
        neutral_score = probs[0].item()

        # We return the difference between Pos and Neg. 
        # This gives a continuous range: -1.0 (Full Bear) to +1.0 (Full Bull)
        sentiment_score = pos_score - neg_score

        # Only return 0.0 if Neutral is truly the dominant label
        if neutral_score > pos_score and neutral_score > neg_score and abs(sentiment_score) < 0.1:
            return 0.0
            
        return sentiment_score
    
    def find_edge_index(self, news_headline, ticker, pyg_data):
        # This logic assumes your node IDs follow a consistent naming convention
        # e.g., 'company_AAPL' and 'news_HASHED_HEADLINE'
        try:
            # 1. Find indices in the NetworkX graph
            u = [n for n, d in self.graph.nodes(data=True) if d.get('label') == news_headline][0]
            v = f"company_{ticker}"
            
            # 2. Map to PyG tensor indices
            node_list = list(self.graph.nodes)
            u_idx, v_idx = node_list.index(u), node_list.index(v)
            
            # 3. Find where this [u, v] exists in edge_index
            mask = (pyg_data.edge_index[0] == u_idx) & (pyg_data.edge_index[1] == v_idx)
            return mask.nonzero(as_tuple=True)[0]
        except:
            return None