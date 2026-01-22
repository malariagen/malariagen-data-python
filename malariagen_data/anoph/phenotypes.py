import pandas as pd
import xarray as xr
from typing import Callable, Optional, List, Any, TYPE_CHECKING
import warnings
import fsspec
from numpydoc_decorator import doc  # type: ignore

from ..util import _check_types
from malariagen_data.anoph import base_params, phenotype_params


class AnophelesPhenotypeData:
    """
    Provides methods for accessing insecticide resistance phenotypic data.
    Inherited by AnophelesDataResource subclasses (e.g., Ag3).
    """

    if TYPE_CHECKING:
        # Type annotations for MyPy
        _url: str
        _fs: fsspec.AbstractFileSystem
        sample_metadata: Callable[..., pd.DataFrame]
        _base_path: str
        _major_version_path: str
        _release_to_path: Callable[[str], str]
        lookup_release: Callable[..., str]
        _prep_sample_sets_param: Callable[..., Any]

        sample_sets: Callable[..., pd.DataFrame]
        snp_calls: Callable[..., Any]
        haplotypes: Callable[..., Any]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_phenotype_data(
        self,
        sample_sets: base_params.sample_sets,
    ) -> pd.DataFrame:
        """
        Load raw phenotypic data from GCS for given sample sets.
        """
        phenotype_dfs = []

        for sample_set in sample_sets:
            try:
                release = self.lookup_release(sample_set=sample_set)
                release_path = self._release_to_path(release)

                phenotype_path = f"{self._base_path}/{release_path}/phenotypes/all/{sample_set}/phenotypes.csv"

                if not self._fs.exists(phenotype_path):
                    warnings.warn(
                        f"Phenotype data file not found for {sample_set} at {phenotype_path}"
                    )
                    continue

                with self._fs.open(phenotype_path, "r") as f:
                    try:
                        df_pheno = pd.read_csv(f, low_memory=False, engine="python")
                    except pd.errors.EmptyDataError:
                        warnings.warn(f"Empty phenotype file for {sample_set}")
                        continue
                    except pd.errors.ParserError as e:
                        warnings.warn(
                            f"Error parsing phenotype file for {sample_set}: {e}"
                        )
                        continue

                df_pheno["sample_set"] = sample_set
                phenotype_dfs.append(df_pheno)

            except Exception as e:
                warnings.warn(
                    f"Unexpected error loading phenotype data for {sample_set}: {e}"
                )
                continue

        if not phenotype_dfs:
            raise ValueError(
                "No phenotype data could be loaded for the specified sample sets"
            )

        # Memory-efficient concatenation
        df_combined = pd.concat(phenotype_dfs, ignore_index=True, sort=False)
        return df_combined

    def _merge_phenotype_with_metadata(
        self,
        df_phenotypes: pd.DataFrame,
        df_sample_metadata: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge phenotypic data with sample metadata.
        """
        # Identify columns present in both, excluding the merge key ("sample_id")
        pheno_cols = set(df_phenotypes.columns)
        meta_cols = set(df_sample_metadata.columns)
        common_cols = (pheno_cols & meta_cols) - {"sample_id"}

        # Drop common columns from metadata *before* merging to avoid suffixes
        df_meta_subset = df_sample_metadata.drop(columns=list(common_cols))

        # Merge phenotype data with the subset of metadata
        df_merged = pd.merge(
            df_phenotypes,
            df_meta_subset,
            on="sample_id",
            how="left",  # Keeping just the phenotype rows
        )

        return df_merged

    def _apply_phenotype_cohort_filtering(
        self,
        df: pd.DataFrame,
        cohort_size: Optional[base_params.cohort_size] = None,
        min_cohort_size: Optional[base_params.min_cohort_size] = None,
        max_cohort_size: Optional[base_params.max_cohort_size] = None,
    ) -> pd.DataFrame:
        """
        Apply cohort size filtering based on insecticide resistance experimental groups.
        """
        if all(
            param is None for param in [cohort_size, min_cohort_size, max_cohort_size]
        ):
            return df

        cohort_keys = ["insecticide", "dose", "location", "country", "sample_set"]
        available_keys = [col for col in cohort_keys if col in df.columns]
        if not available_keys:
            warnings.warn("No suitable columns found for cohort grouping")
            return df

        grouped = df.groupby(available_keys)
        filtered_groups = []

        for name, group in grouped:
            group_size = len(group)
            if min_cohort_size is not None and group_size < min_cohort_size:
                continue
            if max_cohort_size is not None and group_size > max_cohort_size:
                group = group.sample(n=max_cohort_size, random_state=42)
            elif cohort_size is not None and group_size > cohort_size:
                group = group.sample(n=cohort_size, random_state=42)
            elif cohort_size is not None and group_size < cohort_size:
                continue
            filtered_groups.append(group)

        if not filtered_groups:
            warnings.warn("No cohorts met the specified size criteria")
            return pd.DataFrame()

        return pd.concat(filtered_groups, ignore_index=True)

    def _create_phenotype_binary_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Convert phenotype outcomes to binary format(1=alive/resistant, 0=dead/susceptible).
        Handles case-insensitivity and warns about unmapped values.
        """
        if "phenotype" not in df.columns:
            raise ValueError("DataFrame must contain 'phenotype' column")

        phenotype_map = {
            "alive": 1,
            "dead": 0,
            "survived": 1,
            "died": 0,
            "resistant": 1,
            "susceptible": 0,
        }

        # Ensure input is string and lowercase, then map
        phenotype_lower = df["phenotype"].astype(str).str.lower()
        binary_series = phenotype_lower.map(phenotype_map)

        unmapped_mask = binary_series.isna()
        if unmapped_mask.any():
            original_unmapped_values = df.loc[unmapped_mask, "phenotype"].unique()
            warnings.warn(
                f"Unmapped phenotype values found and converted to NaN: {list(original_unmapped_values)}. "
                f"Expected values (case-insensitive): {list(phenotype_map.keys())}"
            )

        if "sample_id" in df.columns:
            binary_series.index = pd.Index(df["sample_id"])
        else:
            warnings.warn(
                "Cannot set index to sample_id as it is missing from the input DataFrame."
            )

        # Convert to appropriate dtype (float64 allows NaN)
        return binary_series.astype(float)

    def _create_phenotype_dataset(
        self,
        df_phenotypes: pd.DataFrame,
        variant_data: Optional[xr.Dataset] = None,
    ) -> xr.Dataset:
        """
        Create xarray Dataset combining phenotypic and genetic variant data.
        """
        df_indexed = df_phenotypes.set_index("sample_id")
        sample_ids = df_indexed.index.values
        phenotype_binary = self._create_phenotype_binary_series(df_phenotypes)

        # Reindex to match df_indexed in case rows were dropped or index was not set previously
        phenotype_binary = phenotype_binary.reindex(sample_ids)

        data_vars = {
            "phenotype_binary": (["samples"], phenotype_binary.values),
            "insecticide": (["samples"], df_indexed["insecticide"].values),
            "dose": (["samples"], df_indexed["dose"].values),
            "phenotype": (["samples"], df_indexed["phenotype"].values),
        }

        # Optional variables
        for var in ["location", "country", "collection_date", "species", "sample_set"]:
            if var in df_indexed.columns:
                data_vars[var] = (["samples"], df_indexed[var].values)

        coords = {"samples": sample_ids}
        ds = xr.Dataset(data_vars, coords=coords)

        if variant_data is not None:
            # Find the sample dimension and coordinate
            sample_dim = None
            sample_coord = None

            # Check for sample dimension
            for dim_name in ["samples", "sample_id"]:
                if dim_name in variant_data.dims:
                    sample_dim = dim_name
                    break

            if sample_dim is None:
                warnings.warn(
                    f"Variant data does not contain 'samples' or 'sample_id' dimension. "
                    f"Available dimensions: {list(variant_data.dims)}"
                )
                return ds

            for coord_name in ["sample_id", "samples"]:
                if coord_name in variant_data.coords:
                    sample_coord = coord_name
                    break

            if sample_coord is None:
                warnings.warn(
                    f"Variant data does not contain 'samples' or 'sample_id' coordinate. "
                    f"Available coordinates: {list(variant_data.coords.keys())}"
                )
                return ds

            # Get variant sample IDs - use the correct coordinate name
            variant_sample_ids = variant_data.coords[sample_coord].values

            # Find common samples
            common_samples = list(set(sample_ids) & set(variant_sample_ids))

            if not common_samples:
                warnings.warn(
                    "No common samples found between phenotype and variant data"
                )
                return ds
            else:
                # Select common samples from phenotype dataset
                ds_common = ds.sel(samples=common_samples)

                # Select common samples from variant dataset
                try:
                    # Use isel with boolean indexing instead of sel
                    sample_mask = pd.Series(variant_sample_ids).isin(common_samples)
                    sample_indices = sample_mask[sample_mask].index.tolist()
                    variant_data_common = variant_data.isel(
                        {sample_dim: sample_indices}
                    )

                    # Rename dimension to "samples" if it\'s not already
                    if sample_dim != "samples":
                        variant_data_common = variant_data_common.rename(
                            {sample_dim: "samples"}
                        )

                    if (
                        sample_coord != "samples"
                        and sample_coord in variant_data_common.coords
                    ):
                        variant_data_common = variant_data_common.rename(
                            {sample_coord: "samples"}
                        )

                    variant_samples_selected = variant_data_common.coords[
                        "samples"
                    ].values

                    ds_common_reordered = ds_common.sel(
                        samples=variant_samples_selected
                    )

                    # Merge the datasets
                    ds = xr.merge([ds_common_reordered, variant_data_common])

                except KeyError as e:
                    warnings.warn(f"Key error in variant data selection: {e}")
                    return ds
                except ValueError as e:
                    warnings.warn(f"Value error in variant data merging: {e}")
                    return ds
                except Exception as e:
                    warnings.warn(
                        f"Unexpected error selecting/merging variant data: {e}"
                    )
                    return ds

        return ds

    @_check_types
    @doc(
        summary="Load phenotypic data from insecticide resistance bioassays.",
        returns=dict(
            df="DataFrame containing phenotype data merged with sample metadata. Includes sample identifiers, phenotypic measurements, and experimental conditions."
        ),
    )
    def phenotype_data(
        self,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        cohort_size: Optional[base_params.cohort_size] = None,
        min_cohort_size: Optional[base_params.min_cohort_size] = None,
        max_cohort_size: Optional[base_params.max_cohort_size] = None,
    ) -> pd.DataFrame:
        """
        Retrieve and merge phenotype data with sample metadata for bioassay analysis.
        """

        # 1. Normalize sample_sets
        sample_sets_norm = self._prep_sample_sets_param(sample_sets=sample_sets)

        # 2. Load raw phenotype data
        df_phenotypes = self._load_phenotype_data(sample_sets=sample_sets_norm)

        if df_phenotypes.empty:
            warnings.warn("No phenotype data found for the specified sample sets")
            return pd.DataFrame()

        # 3. Get sample metadata for all samples that have phenotype data
        try:
            df_metadata = self.sample_metadata(sample_sets=sample_sets_norm)
        except Exception as e:
            warnings.warn(f"Error fetching sample metadata: {e}")
            return pd.DataFrame()

        if df_metadata.empty:
            warnings.warn("No sample metadata found for samples with phenotype data")
            return pd.DataFrame()

        # 4. Merge phenotype data with metadata
        # Filter phenotype data to samples present in metadata before merging
        metadata_sample_ids = set(df_metadata["sample_id"].unique())
        df_phenotypes_filtered = df_phenotypes[
            df_phenotypes["sample_id"].isin(metadata_sample_ids)
        ]

        df_merged = self._merge_phenotype_with_metadata(
            df_phenotypes_filtered, df_metadata
        )

        # 5. Apply user's sample_query if provided
        if sample_query:
            try:
                df_merged = df_merged.query(
                    sample_query, **(sample_query_options or {})
                )
            except Exception as e:
                warnings.warn(f"Error applying sample_query '{sample_query}': {e}")
                return pd.DataFrame()

        # 6. Apply cohort filtering
        df_final = self._apply_phenotype_cohort_filtering(
            df_merged, cohort_size, min_cohort_size, max_cohort_size
        )

        return df_final

    @_check_types
    @doc(
        summary="Combine phenotypic traits with SNP genotype data for GWAS analysis.",
        returns=dict(
            ds="xarray Dataset containing phenotype data and SNP genotype calls for the specified region."
        ),
    )
    def phenotypes_with_snps(
        self,
        region: base_params.region,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        cohort_size: Optional[base_params.cohort_size] = None,
        min_cohort_size: Optional[base_params.min_cohort_size] = None,
        max_cohort_size: Optional[base_params.max_cohort_size] = None,
    ) -> xr.Dataset:
        """Merge phenotypes with SNP calls in a given region for association testing."""

        df_phenotypes = self.phenotype_data(
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
        )

        if df_phenotypes.empty:
            warnings.warn("No phenotype data found for SNP merge.")
            return xr.Dataset()

        sample_ids_with_phenotypes = df_phenotypes["sample_id"].unique().tolist()

        # Fetch SNP calls for the relevant samples
        snp_data = self.snp_calls(
            region=region,
            sample_query=f"sample_id in {sample_ids_with_phenotypes}",
            sample_query_options=sample_query_options,
        )

        ds = self._create_phenotype_dataset(df_phenotypes, snp_data)

        return ds

    @_check_types
    @doc(
        summary="Combine phenotypic traits with haplotype data for extended association analysis.",
        returns=dict(
            ds="xarray Dataset with phenotype and haplotype data for the specified region."
        ),
    )
    def phenotypes_with_haplotypes(
        self,
        region: base_params.region,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        cohort_size: Optional[base_params.cohort_size] = None,
        min_cohort_size: Optional[base_params.min_cohort_size] = None,
        max_cohort_size: Optional[base_params.max_cohort_size] = None,
    ) -> xr.Dataset:
        """Merge phenotypes with haplotype data in a given region for association testing."""

        df_phenotypes = self.phenotype_data(
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
        )

        if df_phenotypes.empty:
            warnings.warn("No phenotype data found for haplotype merge.")
            return xr.Dataset()

        sample_ids_with_phenotypes = df_phenotypes["sample_id"].unique().tolist()

        # Fetch haplotype data for the relevant samples
        haplotype_data = self.haplotypes(
            region=region,
            sample_query=f"sample_id in {sample_ids_with_phenotypes}",
            sample_query_options=sample_query_options,
        )

        ds = self._create_phenotype_dataset(df_phenotypes, haplotype_data)

        return ds

    @_check_types
    @doc(
        summary="List sample sets that contain phenotypic data.",
        returns=dict(sample_sets="List of sample set identifiers with phenotype data."),
    )
    def phenotype_sample_sets(self) -> List[str]:
        """Identify sample sets containing phenotype data."""

        all_sample_sets = self.sample_sets()["sample_set"].tolist()  # type: ignore[operator]
        phenotype_sample_sets = []

        for sample_set in all_sample_sets:
            try:
                release = self.lookup_release(sample_set=sample_set)
                release_path = self._release_to_path(release)

                phenotype_path = f"{self._base_path}/{release_path}/phenotypes/all/{sample_set}/phenotypes.csv"

                if self._fs.exists(phenotype_path):
                    phenotype_sample_sets.append(sample_set)
            except Exception:
                continue

        return phenotype_sample_sets

    @doc(
        summary="Convert phenotype data into binary format for statistical analysis.",
        returns=dict(
            binary="Pandas Series indexed by sample_id with binary classification: 1 for resistant, 0 for susceptible, NaN for unknown."
        ),
    )
    def phenotype_binary(
        self,
        sample_sets: Optional[base_params.sample_sets] = None,
        insecticide: Optional[phenotype_params.insecticide] = None,
        dose: Optional[phenotype_params.dose] = None,
        phenotype: Optional[phenotype_params.phenotype] = None,
        sample_query: Optional[
            base_params.sample_query
        ] = None,  # Allow direct sample_query
        sample_query_options: Optional[base_params.sample_query_options] = None,
        cohort_size: Optional[base_params.cohort_size] = None,
        min_cohort_size: Optional[base_params.min_cohort_size] = None,
        max_cohort_size: Optional[base_params.max_cohort_size] = None,
    ) -> pd.Series:
        """Generate binary phenotypic labels from raw phenotype data."""

        # Build the sample_query string from individual parameters
        query_parts = []
        if insecticide is not None:
            if isinstance(insecticide, list):
                query_parts.append(f"insecticide in {insecticide}")
            else:
                query_parts.append(f"insecticide == '{insecticide}'")
        if dose is not None:
            if isinstance(dose, list):
                query_parts.append(f"dose in {dose}")
            else:
                query_parts.append(f"dose == {dose}")
        if phenotype is not None:
            if isinstance(phenotype, list):
                query_parts.append(f"phenotype in {phenotype}")
            else:
                query_parts.append(f"phenotype == '{phenotype}'")

        # Combine with an existing sample_query if provided
        final_sample_query = sample_query
        if query_parts:
            generated_query = " and ".join(query_parts)
            if final_sample_query:
                final_sample_query = f"({final_sample_query}) and ({generated_query})"
            else:
                final_sample_query = generated_query

        df = self.phenotype_data(
            sample_sets=sample_sets,
            sample_query=final_sample_query,
            sample_query_options=sample_query_options,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
        )

        if df.empty:
            return pd.Series(dtype=float, name="phenotype_binary")

        binary_series = self._create_phenotype_binary_series(df)

        binary_series.name = "phenotype_binary"
        # Ensure the index is correctly set to sample_id
        if "sample_id" in df.columns:
            binary_series.index = pd.Index(df["sample_id"])
        else:
            warnings.warn(
                "Cannot set index to sample_id as it is missing from the DataFrame returned by phenotype_data."
            )

        return binary_series
