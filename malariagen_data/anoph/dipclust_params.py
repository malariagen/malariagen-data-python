"""Parameters for diplotype clustering functions."""

from typing_extensions import Annotated, TypeAlias, Union, Sequence

from .distance_params import distance_metric
from .clustering_params import linkage_method
from .base_params import transcript


linkage_method_default: linkage_method = "complete"

distance_metric_default: distance_metric = "cityblock"

snp_transcript: TypeAlias = Annotated[
    Union[transcript, Sequence[transcript]],
    "A transcript or a list of transcripts",
]
