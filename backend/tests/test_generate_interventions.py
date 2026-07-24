from types import SimpleNamespace
from app.agent.steps.generate_interventions import InterventionSchema, _evaluate_hard_gates

def _candidate(method, cost):
    return InterventionSchema(intervention_type="evaluation_protocol", failure_mechanism="state change failure", proposed_intervention=method, intermediate_effect="separates state changes", measurable_outcome="accuracy", implementation_cost=cost, mechanism_confidence=0.8)

def test_hard_gates_pass_for_evidence_backed_low_cost_intervention():
    gap = SimpleNamespace()
    audit = SimpleNamespace(audit_result="confirmed", remaining_delta="untested state changes")
    contract = SimpleNamespace(allow_model_training=False, allow_large_benchmark=False, gpu_available=False)
    result = _evaluate_hard_gates(gap, audit, ["e1", "e2"], _candidate("Add evaluation protocol", "low cost"), contract)
    assert result["gate_statuses"] == {"evidence": "PASS", "novelty": "PASS", "feasibility": "PASS"}

def test_hard_gates_reject_training_without_permission():
    gap = SimpleNamespace()
    audit = SimpleNamespace(audit_result="confirmed", remaining_delta="untested state changes")
    contract = SimpleNamespace(allow_model_training=False, allow_large_benchmark=False, gpu_available=False)
    result = _evaluate_hard_gates(gap, audit, ["e1", "e2"], _candidate("Fine-tune state model", "requires GPU training"), contract)
    assert result["gate_statuses"]["feasibility"] == "FAIL"
