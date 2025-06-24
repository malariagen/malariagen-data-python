from typing import TypeAlias, Optional, Union, List, Annotated

# Type alias for insecticide parameter
insecticide: TypeAlias = Annotated[
    Optional[Union[str, List[str]]],
    "Insecticide name(s) to filter by. Can be a single insecticide name or a list of names.",
]

# Type alias for dose parameter
dose: TypeAlias = Annotated[
    Optional[Union[float, List[float]]],
    "Insecticide dose(s) to filter by. Can be a single dose value or a list of dose values.",
]

# Type alias for phenotype parameter
phenotype: TypeAlias = Annotated[
    Optional[Union[str, List[str]]],
    "Phenotype outcome(s) to filter by. Can be a single phenotype value (e.g., 'alive', 'dead') or a list of values.",
]
