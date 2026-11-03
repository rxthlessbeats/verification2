"""Jev-based evidence selection for the QEsA condition.

Input is always ``qera.render.selector_view(traj)``; output is a selections
JSONL that ``verification2 run --selector file`` consumes. Labels and hidden
fields are never read.
"""

from verification2.jev.client import JevAnswer, JevClient, JevError, JevResult
from verification2.jev.scorer import CandidateScore, JevScorer
from verification2.jev.select import SelectionResult, select_for_view

__all__ = [
    "CandidateScore",
    "JevAnswer",
    "JevClient",
    "JevError",
    "JevResult",
    "JevScorer",
    "SelectionResult",
    "select_for_view",
]
