"""Parameter definitions for karyotype analysis functions."""


from typing_extensions import Annotated, TypeAlias

inversion_param: TypeAlias = Annotated[
    str,
    "Name of inversion to infer karyotype for.",
]
