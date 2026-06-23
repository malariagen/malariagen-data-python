"""Parameters for Plink converter functions."""

from typing import Mapping, Union

from typing_extensions import Annotated, TypeAlias

overwrite: TypeAlias = Annotated[
    bool,
    """
    A boolean indicating whether a previously written file with the same name ought
    to be overwritten. Default is False.
    """,
]

output_dir: TypeAlias = Annotated[
    str,
    """
    A string indicating the desired output file location.
    """,
]

out: TypeAlias = Annotated[
    str,
    """
    A string specifying the output file path prefix. The PLINK output files
    will be written as ``{output_dir}/{out}.bed``, ``{output_dir}/{out}.bim``,
    and ``{output_dir}/{out}.fam``. If not provided, a default prefix is
    generated from the SNP selection parameters (region, n_snps,
    min_minor_ac, max_missing_an, thin_offset).
    """,
]

phenotypes: TypeAlias = Annotated[
    Mapping[str, Union[int, float]],
    """
    A mapping of sample identifiers to phenotype values. In PLINK format,
    -9 indicates missing phenotype, 1 indicates control (unaffected),
    and 2 indicates case (affected). Continuous values can also be used.
    """,
]
