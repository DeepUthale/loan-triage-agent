"""Wire the nodes into the triage workflow."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from loan_triage.graph.nodes import TriageNodes
from loan_triage.graph.state import TriageState


def build_graph(nodes: TriageNodes, checkpointer: BaseCheckpointSaver[Any]) -> Any:
    graph: Any = StateGraph(TriageState)

    graph.add_node("ingest", nodes.ingest)
    graph.add_node("classify_document", nodes.classify_document)
    graph.add_node("plan_extraction", nodes.plan_extraction)
    graph.add_node("extract_document", nodes.extract_document)
    graph.add_node("assess_quality", nodes.assess_quality)
    graph.add_node("reextract_document", nodes.reextract_document)
    graph.add_node("validate", nodes.validate)
    graph.add_node("assess_risk", nodes.assess_risk)
    graph.add_node("route", nodes.route)
    graph.add_node("prepare_review", nodes.prepare_review)
    graph.add_node("human_review", nodes.human_review)
    graph.add_node("apply_corrections", nodes.apply_corrections)
    graph.add_node("request_documents", nodes.request_documents)
    graph.add_node("await_documents", nodes.await_documents)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "ingest")
    graph.add_conditional_edges(
        "ingest", nodes.dispatch_classification, ["classify_document", "assess_quality"]
    )
    graph.add_edge("classify_document", "plan_extraction")
    graph.add_conditional_edges(
        "plan_extraction", nodes.dispatch_extraction, ["extract_document", "assess_quality"]
    )
    graph.add_edge("extract_document", "assess_quality")
    graph.add_conditional_edges(
        "assess_quality", nodes.dispatch_reextraction, ["reextract_document", "validate"]
    )
    graph.add_edge("reextract_document", "assess_quality")
    graph.add_edge("validate", "assess_risk")
    graph.add_edge("assess_risk", "route")
    graph.add_conditional_edges(
        "route", nodes.next_after_route, ["finalize", "request_documents", "prepare_review"]
    )
    graph.add_edge("prepare_review", "human_review")
    graph.add_conditional_edges(
        "human_review",
        nodes.next_after_review,
        ["apply_corrections", "request_documents", "finalize"],
    )
    graph.add_edge("apply_corrections", "validate")
    graph.add_edge("request_documents", "await_documents")
    graph.add_edge("await_documents", "ingest")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)
