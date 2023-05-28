from typing import Mapping, Tuple

from typing_extensions import Annotated, TypeAlias

aims: TypeAlias = Annotated[
    str,
    """
    Identifier for a set of ancestry informative markers to use. For
    possible values see the `aim_ids` property.
    """,
]

aim_ids: TypeAlias = Annotated[
    Tuple[aims, ...], """Tuple of identifiers for AIM sets available."""
]

palette = Annotated[
    Tuple[str, str, str, str],
    """
    4-tuple of colors for AIM genotypes, in the order: missing, hom
    taxon 1, het, hom taxon 2.
    """,
]

aim_palettes = Annotated[
    Mapping[aims, palette],
    """
    Mapping from AIM IDs to color palettes. I.e., provides a genotype
    color palette for each set of AIMs.
    """,
]
