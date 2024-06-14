"""Parameters for diplotype clustering functions."""

from .diplotype_distance_params import distance_metric
from .clustering_params import linkage_method


linkage_method_default: linkage_method = "complete"

distance_metric_default: distance_metric = "cityblock"
