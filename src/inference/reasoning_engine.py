import os
import torch
from google import genai # New SDK
from google.genai import types
from dotenv import load_dotenv
from src.graph.financial_kg import DynamicFinancialKG

load_dotenv()

class LookUPReasoningEngine:
    def __init__(self, kg, model_path):
        self.kg = kg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kg.gnn_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.kg.gnn_model.eval()

    def get_causal_drivers(self, ticker, top_k=3):
        pyg_data = self.kg.to_pyg_data()
        with torch.no_grad():
            _, (edge_index, alpha) = self.kg.gnn_model(
                pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr, return_attention=True
            )
        
        target_idx = self.kg.get_node_index(ticker)
        if target_idx is None: return []

        mask = (edge_index[1] == target_idx)
        news_alphas = alpha[mask].mean(dim=-1).cpu().numpy()
        relevant_edges = edge_index[:, mask].cpu().numpy()

        drivers = []
        node_list = list(self.kg.graph.nodes)
        for i, weight in enumerate(news_alphas):
            source_idx = relevant_edges[0, i]
            node_id = node_list[source_idx]
            node_data = self.kg.graph.nodes[node_id]
            if node_data.get('type') == 'news':
                drivers.append({
                    'headline': node_data.get('label', 'Unknown Headline'),
                    'impact_score': float(weight),
                    'sentiment': node_data.get('sentiment', 0)
                })
        return sorted(drivers, key=lambda x: x['impact_score'], reverse=True)[:top_k]

class LookUPReporter:
    def __init__(self, kg, model_path):
        # 1. New SDK Client Initialization
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.engine = LookUPReasoningEngine(kg, model_path)

    def generate_report(self, ticker):
        drivers = self.engine.get_causal_drivers(ticker)
        if not drivers:
            return "No causal drivers found."

        # Creating a structured context for the LLM
        context = "\n".join([f"- Headline: {d['headline']} (Weight: {d['impact_score']:.4f}, Sentiment: {d['sentiment']:.2f})" for d in drivers])
        
        prompt = f"""
        You are the LookUP Financial Analyst. Use the following GNN-prioritized causal drivers 
        to explain the recent price movement of {ticker}:
        
        {context}
        
        Provide a concise, data-driven explanation and a 'Causal Confidence Score'.
        """
        
        # Using gemini-2.0-flash for high speed and better availability
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"❌ Gemini API Error: {str(e)}"

# --- Execution ---
if __name__ == "__main__":
    # Note: Ensure your KG is built/populated before calling this
    reporter = LookUPReporter("NVDA")
    report = reporter.generate_report()
    
    print("\n" + "="*40)
    print(f"📈 LOOKUP CAUSAL REPORT: {reporter.ticker}")
    print("="*40 + "\n")
    print(report)