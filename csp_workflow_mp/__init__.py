"""
csp_workflow_mp: Formula-to-Structure Generation via Space-Group-Guided Template Retrieval.

Trained on Materials Project data (CC-BY 4.0). For the AWA-trained version see csp_workflow/.
"""

__version__ = "0.1.0"

from csp_workflow_mp.role_grouping import RoleGrouping
from csp_workflow_mp.symmetry_filter import (
    allowed_pearson_set,
    is_valid_combination,
    sg_to_pearson_prefix,
    filter_compatible_pairs,
)
from csp_workflow_mp.substitution_engine import SubstitutionEngine
from csp_workflow_mp.descriptor import compute_periodic_descriptors
from csp_workflow_mp.retriever import TemplatePool
from csp_workflow_mp.classifier import predict_top_k_space_groups

__all__ = [
    "RoleGrouping",
    "SubstitutionEngine",
    "TemplatePool",
    "allowed_pearson_set",
    "is_valid_combination",
    "sg_to_pearson_prefix",
    "filter_compatible_pairs",
    "compute_periodic_descriptors",
    "predict_top_k_space_groups",
]
