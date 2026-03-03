from .state import LOOKUPState

def competitor_agent(state: LOOKUPState) -> dict:
    """
    Competitor agent: refines competitor candidates based on additional signals.
    For now, just passes through the candidates from GNN.
    """
    return {
        "competitor_candidates": state.get("competitor_candidates", [])
    }