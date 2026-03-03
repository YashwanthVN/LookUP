from .state import LOOKUPState
from src.inference.reasoning_engine import LookUPReporter
from src.kg_holder import get_kg
import os

def writer_agent(state: LOOKUPState) -> dict:
    kg = get_kg()
    primary_ticker = state["tickers"][0]
    
    # Path logic
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    model_path = os.path.join(project_root, "calibrated_gnn_reasoning.pt")
    
    reporter = LookUPReporter(kg, model_path)
    report_text = reporter.generate_report(primary_ticker)
    
    # Format the GNN Specialist Insights section
    insights = "\n".join(state.get("gnn_evidence", []))
    comps = ", ".join(state.get("competitor_candidates", []))
    
    final_output = f"{report_text}\n\n"
    final_output += "---\n**🧠 GNN Specialist Insights**\n"
    final_output += insights if insights else "No specific GNN drivers identified in state."
    
    if comps:
        final_output += f"\n\n**🏢 Top Competitor Candidates:** {comps}"
        
    return {"final_report": final_output}