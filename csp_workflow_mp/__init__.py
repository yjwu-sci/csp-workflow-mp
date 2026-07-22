"""
csp_workflow_mp: Formula-to-Structure Generation via Space-Group-Guided Template Retrieval.

Trained on Materials Project data (CC-BY 4.0). Companion code to
Wu Y.-J. & Xu Y. (manuscript in preparation).
"""

__version__ = "0.1.0"

from csp_workflow_mp.role_grouping import RoleGrouping
from csp_workflow_mp.substitution_engine import SubstitutionEngine
from csp_workflow_mp.descriptor import compute_periodic_descriptors
from csp_workflow_mp.retriever import TemplatePool
from csp_workflow_mp.classifier import predict_top_k_space_groups
from csp_workflow_mp.predict import predict_from_formula, PredictionResult

__all__ = [
    "RoleGrouping",
    "SubstitutionEngine",
    "TemplatePool",
    "compute_periodic_descriptors",
    "predict_top_k_space_groups",
    "predict_from_formula",
    "PredictionResult",
]
