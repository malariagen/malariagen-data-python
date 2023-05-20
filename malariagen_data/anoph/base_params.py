"""General parameters common to many functions in the public API."""

from typing import Final, List, Mapping, Optional, Sequence, Tuple, Union

from typing_extensions import Annotated, TypeAlias

from ..util import region_param_type, single_region_param_type

contig: TypeAlias = Annotated[
    str,
    """
    Reference genome contig name. See the `contigs` property for valid contig
    names.
    """,
]

single_region: TypeAlias = Annotated[
    single_region_param_type,
    """
    Region of the reference genome. Can be a contig name, region string
    (formatted like "{contig}:{start}-{end}"), or identifier of a genome
    feature such as a gene or transcript.
    """,
]

region: TypeAlias = Annotated[
    region_param_type,
    """
    Region of the reference genome. Can be a contig name, region string
    (formatted like "{contig}:{start}-{end}"), or identifier of a genome
    feature such as a gene or transcript. Can also be a sequence (e.g., list)
    of regions.
    """,
]

release: TypeAlias = Annotated[
    Union[str, Sequence[str]],
    "Release version identifier.",
]

sample_set: TypeAlias = Annotated[
    str,
    "Sample set identifier.",
]

sample_sets: TypeAlias = Annotated[
    Union[Sequence[str], str],
    """
    List of sample sets and/or releases. Can also be a single sample set or
    release.
    """,
]

sample_query: TypeAlias = Annotated[
    str,
    """
    A pandas query string to be evaluated against the sample metadata, to
    select samples to be included in the returned data.
    """,
]

sample_indices: TypeAlias = Annotated[
    List[int],
    """
    Advanced usage parameter. A list of indices of samples to select,
    corresponding to the order in which the samples are found within the
    sample metadata. Either provide this parameter or sample_query, not
    both.
    """,
]


def validate_sample_selection_params(
    *,
    sample_query: Optional[sample_query],
    sample_indices: Optional[sample_indices],
):
    if sample_query is not None and sample_indices is not None:
        raise ValueError(
            "Please provide either sample_query or sample_indices, not both."
        )


cohort1_query: TypeAlias = Annotated[
    str,
    """
    A pandas query string to be evaluated against the sample metadata,
    to select samples for the first cohort.
    """,
]

cohort2_query: TypeAlias = Annotated[
    str,
    """
    A pandas query string to be evaluated against the sample metadata,
    to select samples for the second cohort.
    """,
]

site_mask: TypeAlias = Annotated[
    str,
    """
    Which site filters mask to apply. See the `site_mask_ids` property for
    available values.
    """,
]

site_class: TypeAlias = Annotated[
    str,
    """
    Select sites belonging to one of the following classes: CDS_DEG_4,
    (4-fold degenerate coding sites), CDS_DEG_2_SIMPLE (2-fold simple
    degenerate coding sites), CDS_DEG_0 (non-degenerate coding sites),
    INTRON_SHORT (introns shorter than 100 bp), INTRON_LONG (introns
    longer than 200 bp), INTRON_SPLICE_5PRIME (intron within 2 bp of
    5' splice site), INTRON_SPLICE_3PRIME (intron within 2 bp of 3'
    splice site), UTR_5PRIME (5' untranslated region), UTR_3PRIME (3'
    untranslated region), INTERGENIC (intergenic, more than 10 kbp from
    a gene).
    """,
]

cohort_size: TypeAlias = Annotated[
    int,
    """
    Randomly down-sample to this value if the number of samples in the
    cohort is greater. Raise an error if the number of samples is less
    than this value.
    """,
]

min_cohort_size: TypeAlias = Annotated[
    int,
    """
    Minimum cohort size. Raise an error if the number of samples is
    less than this value.
    """,
]

max_cohort_size: TypeAlias = Annotated[
    int,
    """
    Randomly down-sample to this value if the number of samples in the
    cohort is greater.
    """,
]

random_seed: TypeAlias = Annotated[
    int,
    "Random seed used for reproducible down-sampling.",
]

transcript: TypeAlias = Annotated[
    str,
    "Gene transcript identifier.",
]

cohort: TypeAlias = Annotated[
    Union[str, Tuple[str, str]],
    """
    Either a string giving one of the predefined cohort labels, or a
    pair of strings giving a custom cohort label and a sample query.
    """,
]

cohorts: TypeAlias = Annotated[
    Union[str, Mapping[str, str]],
    """
    Either a string giving the name of a predefined cohort set (e.g.,
    "admin1_month") or a dict mapping custom cohort labels to sample
    queries.
    """,
]

n_jack: TypeAlias = Annotated[
    int,
    """
    Number of blocks to divide the data into for the block jackknife
    estimation of confidence intervals. N.B., larger is not necessarily
    better.
    """,
]

confidence_level: TypeAlias = Annotated[
    float,
    """
    Confidence level to use for confidence interval calculation. E.g., 0.95
    means 95% confidence interval.
    """,
]

field: TypeAlias = Annotated[str, "Name of array or column to access."]

inline_array: TypeAlias = Annotated[
    bool,
    "Passed through to dask `from_array()`.",
]

inline_array_default: inline_array = True

chunks: TypeAlias = Annotated[
    Union[str, Tuple[int, ...]],
    """
    If 'auto' let dask decide chunk size. If 'native' use native zarr
    chunks. Also, can be a target size, e.g., '200 MiB', or a tuple of
    integers.
    """,
]

chunks_default: chunks = "native"

gff_attributes: TypeAlias = Annotated[
    Optional[Union[Sequence[str], str]],
    """
    GFF attribute keys to unpack into dataframe columns. Provide "*" to unpack all
    attributes.
    """,
]

DEFAULT: Final[str] = "default"
