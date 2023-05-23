from typing_extensions import Annotated, TypeAlias

aims: TypeAlias = Annotated[
    str,
    """
    Which set of ancestry informative markers to use.
    """,
]
