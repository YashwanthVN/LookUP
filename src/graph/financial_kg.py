import os
import torch
import networkx as nx
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

from torch_geometric.data import Data
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel

# Custom internal imports
from src.graph.gnn_rag import FinancialReasoningGNN
from src.streaming.news_fetcher import UnifiedNewsFetcher

class DynamicFinancialKG:
    """
    Builds and maintains a Dynamic Financial Knowledge Graph integrating 
    fundamental metrics, competitor influence, and real-time news sentiment.
    """
    METRIC_TYPES = ["revenue", "ebitda", "netProfitMargin", "eps", "pe_ratio"]

    def __init__(self, quarters: int = 4, api_key: Optional[str] = None):
        from src.streaming.financial_client import FMPClient
        
        self.client = FMPClient(api_key)
        self.graph = nx.MultiDiGraph()
        self.quarters = quarters

        # Environment-Agnostic Pathing
        project_root = Path(__file__).resolve().parent.parent.parent
        adapter_path = project_root / "notebooks" / "finbert_sentiment_lora_final"

        base_model_name = "yiyanghkust/finbert-tone"
        self.tokenizer = BertTokenizer.from_pretrained(base_model_name)
        base_model = BertForSequenceClassification.from_pretrained(base_model_name)

        try:
            self.sentiment_model = PeftModel.from_pretrained(base_model, adapter_path)
            print("✅ Success: Loaded LoRA Sentiment Model.")
        except Exception:
            print("⚠️ Warning: LoRA not found. Using vanilla FinBERT.")
            self.sentiment_model = base_model

        self.gnn_model = FinancialReasoningGNN(
            in_channels=3, 
            edge_channels=1, 
            hidden=64, 
            out=32
        )
        print("🧠 Reasoning GNN initialized.")

    def build_for_tickers(self, tickers: List[str]):
        self.graph.clear()
        
        # Graceful fail for Commodities (Gold/Silver)
        try:
            bulk_data = self.client.get_bulk_data(tickers, self.quarters)
        except Exception:
            bulk_data = {"profiles": {}, "financials": {}}

        for ticker in tickers:
            prof = bulk_data["profiles"].get(ticker, {})
            sector = prof.get("sector", "Commodity/Other")
            self.graph.add_node(
                f"company_{ticker}", 
                type="company", 
                sector=sector, 
                feat=[1.0, 0.0, 0.0]
            )

            # Add Time Nodes
            for q in range(self.quarters):
                date_node = (datetime.now() - timedelta(days=90 * q)).strftime('%Y-%m')
                self.graph.add_node(f"time_{date_node}", type="time", feat=[0.0, 0.0, 1.0])

            # Build System 1 (Financials)
            fin_df = bulk_data["financials"].get(ticker, pd.DataFrame())
            for m in self.METRIC_TYPES:
                self.graph.add_node(f"metric_{m}", type="metric", feat=[0.0, 1.0, 0.0])
                if not fin_df.empty and m in fin_df.columns:
                    for _, row in fin_df.iterrows():
                        self._add_metric_edges(ticker, m, str(row['date'])[:7], row[m])

        self._add_competitor_edges(tickers)
        self._normalize_metrics()

    def _add_metric_edges(self, ticker, metric, date_prefix, val):
        comp_node = f"company_{ticker}"
        met_node = f"metric_{metric}"
        time_node = f"time_{date_prefix}"
        
        clean_val = float(val) if val and np.isfinite(val) else 0.0

        # Restore Temporal Edges
        self.graph.add_edge(
            comp_node, met_node, relation="REPORTED", value=clean_val, is_metric=True
        )
        if time_node in self.graph:
            self.graph.add_edge(met_node, time_node, relation="VALID_AT")

    def _add_competitor_edges(self, tickers):
        for ticker in tickers:
            u_node = f"company_{ticker}"
            u_sector = self.graph.nodes[u_node].get('sector')
            peers = [
                n for n, d in self.graph.nodes(data=True) 
                if d.get('sector') == u_sector and n != u_node
            ]

            for peer in peers:
                # Uses FMPClient's historical price method
                corr = self._get_correlation_weight(ticker, peer.replace("company_", ""))
                self.graph.add_edge(
                    peer, u_node, relation="COMPETITOR_INFLUENCE", value=float(corr)
                )

    def _get_correlation_weight(self, t1, t2):
        """
        Calculates Pearson correlation between two tickers with length safety checks.
        """
        try:
            p1 = self.client.get_historical_prices(t1)
            p2 = self.client.get_historical_prices(t2)
            
            # --- FIX: Minimum length requirement for valid correlation ---
            if len(p1) < 5 or len(p2) < 5:
                return 0.5  # Neutral fallback
            
            # Ensure we are comparing arrays of the same length
            min_len = min(len(p1), len(p2))
            
            # Calculate correlation on the most recent overlapping window
            correlation = np.corrcoef(p1[-min_len:], p2[-min_len:])[0, 1]
            
            # Handle potential NaNs from np.corrcoef (e.g., if one series is constant)
            if np.isnan(correlation):
                return 0.5
                
            return max(0.1, correlation)
            
        except Exception as e:
            print(f"⚠️ Correlation calculation failed for {t1}-{t2}: {e}")
            return 0.5

    def _normalize_metrics(self):
        for m_type in self.METRIC_TYPES:
            edges = [
                d for u, v, d in self.graph.edges(data=True) 
                if f"metric_{m_type}" in v and d.get('is_metric')
            ]
            if not edges:
                continue
            
            vals = [e['value'] for e in edges]
            mean, std = np.mean(vals), np.std(vals)
            for e in edges:
                e['value'] = (e['value'] - mean) / (std if std > 0 else 1.0)

    def add_news_event(self, ticker, sentiment, magnitude, headline=None):
        news_id = f"news_{ticker}_{hash(headline) % 10000}"
        self.graph.add_node(
            news_id, 
            type="news", 
            feat=[0.0, 0.0, 1.0],
            sentiment=float(sentiment), 
            label=str(headline)
        )
        self.graph.add_edge(news_id, f"company_{ticker}", relation="IMPACTS", value=magnitude)

    def inject_real_time_news(self, ticker: str):
        fetcher = UnifiedNewsFetcher()
        items = fetcher.fetch(ticker, limit=12)
        
        if not items:
            print(f"⚠️ No news items found for {ticker}")
            return

        headlines = [item['headline'] for item in items if item.get('headline')]
        if not headlines:
            print("⚠️ No valid headlines to analyze.")
            return

        # Tokenize for batch sentiment
        inputs = self.tokenizer(
            headlines, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        device = next(self.sentiment_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        self.sentiment_model.eval()
        with torch.no_grad():
            logits = self.sentiment_model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

        hints = fetcher.TEMPLATES.get(ticker.upper(), fetcher.TEMPLATES["DEFAULT"])
        critical_triggers = ["strike", "war", "killed", "attack", "crash", "surge"]

        for idx, item in enumerate(items):
            headline = item['headline']
            if not headline:
                continue

            # Sentiment score (FinBERT: 0=neutral, 1=positive, 2=negative)
            score = probs[idx][1].item() - probs[idx][2].item()
            headline_lower = headline.lower()

            # Base magnitude
            magnitude = abs(score) * 2.0 + 0.5

            # Boosting
            match_count = sum(1 for hint in hints if hint.lower() in headline_lower)
            is_critical = any(trigger in headline_lower for trigger in critical_triggers)

            if is_critical:
                magnitude *= 4.0
            elif match_count > 0:
                magnitude *= (1.0 + (0.5 * match_count))

            self.add_news_event(ticker, score, magnitude, headline)

        print(f"✅ Injected {len(items)} news events for {ticker}")

    def get_node_index(self, label: str):
        node_list = list(self.graph.nodes)
        try:
            if label in self.METRIC_TYPES:
                return node_list.index(f"metric_{label}")
            
            if label in node_list:
                return node_list.index(label)
            
            prefixed = f"company_{label}"
            if prefixed in node_list:
                return node_list.index(prefixed)
                
            return None
        except ValueError:
            return None

    def to_pyg_data(self) -> Data:
        node_list = list(self.graph.nodes)
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        
        x = torch.tensor(
            [self.graph.nodes[n].get('feat', [0, 0, 0]) for n in node_list], 
            dtype=torch.float
        )
        
        edge_index, edge_attr = [], []
        for u, v, d in self.graph.edges(data=True):
            if u in node_to_idx and v in node_to_idx:
                edge_index.append([node_to_idx[u], node_to_idx[v]])
                edge_attr.append([float(d.get('value', 1.0))])
        
        if not edge_index:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)