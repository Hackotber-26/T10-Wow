"""
RootSight - Evaluation Metrics Module
Computes Top-1, Top-3, and Mean Reciprocal Rank (MRR) accuracy metrics
with strict accounting of missing/unranked cases in the denominator.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


@dataclass
class IncidentEvaluationResult:
    incident_id: str
    ground_truth: str
    predicted_top1: str
    predicted_top3: List[str]
    is_top1_correct: bool
    is_top3_correct: bool
    rank_of_truth: Optional[int]
    reciprocal_rank: float
    confidence_level: str
    method: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkSummary:
    method: str
    dataset_name: str
    total_incidents: int
    evaluated_incidents: int
    top1_correct_count: int
    top3_correct_count: int
    top1_accuracy: float
    top3_accuracy: float
    mrr: float
    results: List[IncidentEvaluationResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "dataset_name": self.dataset_name,
            "total_incidents": self.total_incidents,
            "evaluated_incidents": self.evaluated_incidents,
            "top1_correct_count": self.top1_correct_count,
            "top3_correct_count": self.top3_correct_count,
            "top1_accuracy": round(self.top1_accuracy, 4),
            "top3_accuracy": round(self.top3_accuracy, 4),
            "mrr": round(self.mrr, 4),
            "results": [r.to_dict() for r in self.results],
        }


def compute_benchmark_metrics(
    eval_results: List[IncidentEvaluationResult],
    method_name: str,
    dataset_name: str,
    total_expected: Optional[int] = None,
) -> BenchmarkSummary:
    """
    Computes overall benchmark statistics. Never drops failed cases from denominator.
    """
    total = total_expected if total_expected is not None else len(eval_results)
    if total == 0:
        return BenchmarkSummary(
            method=method_name,
            dataset_name=dataset_name,
            total_incidents=0,
            evaluated_incidents=0,
            top1_correct_count=0,
            top3_correct_count=0,
            top1_accuracy=0.0,
            top3_accuracy=0.0,
            mrr=0.0,
            results=[],
        )

    top1_count = sum(1 for r in eval_results if r.is_top1_correct)
    top3_count = sum(1 for r in eval_results if r.is_top3_correct)
    total_rr = sum(r.reciprocal_rank for r in eval_results)

    return BenchmarkSummary(
        method=method_name,
        dataset_name=dataset_name,
        total_incidents=total,
        evaluated_incidents=len(eval_results),
        top1_correct_count=top1_count,
        top3_correct_count=top3_count,
        top1_accuracy=top1_count / total,
        top3_accuracy=top3_count / total,
        mrr=total_rr / total,
        results=eval_results,
    )
