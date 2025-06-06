"""
Type aliases for phenotype-related parameters.
"""
from typing import TypeAlias, Optional, Union, List

# Type alias for insecticide parameter
insecticide: TypeAlias = Optional[Union[str, List[str]]]

# Type alias for dose parameter
dose: TypeAlias = Optional[Union[float, List[float]]]

# Type alias for phenotype parameter
phenotype: TypeAlias = Optional[Union[str, List[str]]]
