"""Parameter definitions for karyotype analysis functions."""

from typing import Literal

from typing_extensions import Annotated, TypeAlias

inversion_param: TypeAlias = Annotated[
    Literal["2La", "2Rb", "2Rc_gam", "2Rc_col", "2Rd", "2Rj"],
    "Name of inversion to infer karyotype for.",
]
