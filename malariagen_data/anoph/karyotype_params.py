"""Parameter definitions for karyotype analysis functions."""
from typing import Union, Sequence

from typing_extensions import Annotated, TypeAlias

inversion_param: TypeAlias = Annotated[
    str,
    "Name of inversion to infer karyotype for.",
]

inversions_param: TypeAlias = Annotated[
    Union[Sequence[inversion_param], inversion_param],
    "Names of inversion to infer karyotype for.",
]
