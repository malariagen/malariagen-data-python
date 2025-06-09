import pandas as pd
import xarray as xr
from typing import Callable, Optional, List, Any
import warnings
import fsspec
from malariagen_data.anoph import base_params, phenotype_params


class AnophelesPhenotypeData:

    """
    Provides methods for accessing insecticide resistance phenotypic data.
    Inherited by AnophelesDataResource subclasses (e.g., Ag3).
    """

    _url: str
    _fs: fsspec.AbstractFileSystem
    sample_metadata: Callable[..., pd.DataFrame]
    sample_sets: list[str]
    snp_calls: Callable[..., Any]
    _prep_sample_sets_param: Callable[..., Any]
    haplotypes: Callable[..., Any]

    def __init__(
        self,
        url: str,
        fs: fsspec.AbstractFileSystem,
        sample_metadata: Callable[..., pd.DataFrame],
        sample_sets: list[str],
        snp_calls: Callable[..., Any],
        prep_sample_sets_param: Callable[..., Any],
        haplotypes: Callable[..., Any],
    ):
        """
        Initialize the AnophelesPhenotypeData class.

        Parameters
        ----------
        url : str
            Base URL for accessing phenotype data.
        fs : fsspec.AbstractFileSystem
            File system interface for accessing remote data.
        sample_metadata : callable
            Function to retrieve sample metadata.
        sample_sets : list of str
            List of available sample sets.
        snp_calls : callable
            Function to retrieve SNP calls.
        prep_sample_sets_param : callable
            Function to prepare sample sets parameter.
        haplotypes : callable
            Function to retrieve haplotype data.
        """
        self._url = url
        self._fs = fs
        self.sample_metadata = sample_metadata
        self.sample_sets = sample_sets
        self.snp_calls = snp_calls
        self._prep_sample_sets_param = prep_sample_sets_param
        self.haplotypes = haplotypes

    def _load_phenotype_data(
        self,
        sample_sets: base_params.sample_sets,
        insecticide: phenotype_params.insecticide,
        dose: phenotype_params.dose = None,
        phenotype: phenotype_params.phenotype = None,
    ) -> pd.DataFrame:
        """
        Load raw phenotypic data from GCS for given sample sets.
        """
        phenotype_dfs = []
        base_phenotype_path = f"{self._url}v3.2/phenotypes/all"

        for sample_set in sample_sets:
            phenotype_path = f"{base_phenotype_path}/{sample_set}/phenotypes.csv"
            try:
                if not self._fs.exists(phenotype_path):
                    warnings.warn(
                        f"Phenotype data file not found for {sample_set} at {phenotype_path}"
                    )
                    continue

                with self._fs.open(phenotype_path, "r") as f:
                    df_pheno = pd.read_csv(f)

                df_pheno["sample_set"] = sample_set
                phenotype_dfs.append(df_pheno)

            except FileNotFoundError:
                warnings.warn(
                    f"Phenotype data file not found for {sample_set} at {phenotype_path}"
                )
                continue
            except Exception as e:
                warnings.warn(
                    f"Unexpected error loading phenotype data for {sample_set} from {phenotype_path}: {e}"
                )
                continue

        if not phenotype_dfs:
            raise ValueError(
                "No phenotype data could be loaded for the specified sample sets"
            )

        df_combined = pd.concat(phenotype_dfs, ignore_index=True)

        # Apply simple filters
        if insecticide is not None:
            df_combined = df_combined[df_combined["insecticide"].isin(insecticide)]

        if dose is not None:
            df_combined = df_combined[
                df_combined["dose"].isin(dose if isinstance(dose, list) else [dose])
            ]

        if phenotype is not None:
            df_combined = df_combined[df_combined["phenotype"].isin(phenotype)]

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

        # Convert to appropriate dtype (float64 allows NaN, int64 is also an option)
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
            if "samples" not in variant_data.dims:
                warnings.warn(
                    "Variant data does not contain 'samples' dimension, cannot merge."
                )
            else:
                common_samples = list(
                    set(sample_ids) & set(variant_data.coords["samples"].values)
                )
                if not common_samples:
                    warnings.warn(
                        "No common samples found between phenotype and variant data"
                    )
                else:
                    ds_common = ds.sel(samples=common_samples)
                    variant_data_common = variant_data.sel(samples=common_samples)
                    ds = xr.merge([ds_common, variant_data_common])

        return ds

    def _validate_phenotype_parameters(
        self,
        insecticide: phenotype_params.insecticide = None,
        dose: phenotype_params.dose = None,
        phenotype: phenotype_params.phenotype = None,
    ) -> tuple:
        """
        Validate and normalize phenotype-specific parameters.
        """
        # Insecticide normalization
        if isinstance(insecticide, str):
            insecticide = [insecticide]
        elif insecticide is not None and not isinstance(insecticide, list):
            raise TypeError("insecticide must be str, list of str, or None")

        # Dose normalization
        if isinstance(dose, (int, float)):
            dose = [float(dose)]
        elif dose is not None:
            if isinstance(dose, list):
                dose = [float(d) for d in dose]
            else:
                raise TypeError("dose must be numeric, list of numeric, or None")

        # Phenotype normalization
        if isinstance(phenotype, str):
            phenotype = [phenotype]
        elif phenotype is not None and not isinstance(phenotype, list):
            raise TypeError("phenotype must be str, list of str, or None")

        return insecticide, dose, phenotype

    def phenotype_data(
        self,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        insecticide: phenotype_params.insecticide = None,
        dose: phenotype_params.dose = None,
        phenotype: phenotype_params.phenotype = None,
        cohort_size: Optional[base_params.cohort_size] = None,
        min_cohort_size: Optional[base_params.min_cohort_size] = None,
        max_cohort_size: Optional[base_params.max_cohort_size] = None,
    ) -> pd.DataFrame:
        """
        Load phenotypic data from insecticide resistance bioassays.
        """
        # 1. Normalize sample_sets
        sample_sets_norm = self._prep_sample_sets_param(sample_sets=sample_sets)

        # 2. Validate parameters
        (
            insecticide_norm,
            dose_norm,
            phenotype_norm,
        ) = self._validate_phenotype_parameters(insecticide, dose, phenotype)

        # 3. Load raw phenotype data
        df_phenotypes = self._load_phenotype_data(
            sample_sets=sample_sets_norm,
            insecticide=insecticide_norm,
            dose=dose_norm,
            phenotype=phenotype_norm,
        )

        # 4. Get metadata for those samples
        sample_ids_with_phenotypes = df_phenotypes["sample_id"].unique().tolist()
        if not sample_ids_with_phenotypes:
            warnings.warn(
                "No samples found in loaded phenotype data, cannot fetch metadata."
            )
            return df_phenotypes

        # Fetch metadata only for samples that have phenotype data
        df_sample_metadata = self.sample_metadata(
            sample_sets=sample_sets,
            sample_query=f"sample_id in {sample_ids_with_phenotypes}",
        )

        # 5. Merge phenotype + metadata
        df_merged: pd.DataFrame = self._merge_phenotype_with_metadata(
            df_phenotypes, df_sample_metadata
        )

        # 6. Apply sample_query if provided
        if sample_query is not None:
            try:
                df_merged = self.sample_metadata(
                    sample_query=sample_query,
                    sample_query_options=sample_query_options,
                    df=df_merged,  # Pass the dataframe to filter
                )
            except TypeError as e:
                if "unexpected keyword argument 'df'" in str(e):
                    warnings.warn(
                        "Cannot apply sample_query to merged data; base sample_metadata might not support filtering an existing DataFrame."
                    )
                else:
                    raise e
            except Exception as e:
                warnings.warn(f"Error applying sample_query to merged data: {e}")

        # 7. Apply cohort filtering
        df_final = self._apply_phenotype_cohort_filtering(
            df_merged, cohort_size, min_cohort_size, max_cohort_size
        )

        return df_final

    def phenotype_binary(
        self,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        insecticide: phenotype_params.insecticide = None,
        dose: phenotype_params.dose = None,
        phenotype: phenotype_params.phenotype = None,
        cohort_size: Optional[base_params.cohort_size] = None,
        min_cohort_size: Optional[base_params.min_cohort_size] = None,
        max_cohort_size: Optional[base_params.max_cohort_size] = None,
    ) -> pd.Series:
        """
        Load phenotypic data as binary outcomes (1=alive/resistant, 0=dead/susceptible, NaN=unknown).
        Returns a pandas Series indexed by sample_id.
        """

        df = self.phenotype_data(
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            insecticide=insecticide,
            dose=dose,
            phenotype=phenotype,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
        )
        if df.empty:
            return pd.Series(dtype=float, name="phenotype_binary")

        binary_series = self._create_phenotype_binary_series(df)

        binary_series.name = "phenotype_binary"

        binary_series.index = df["sample_id"].index

        return binary_series

    def phenotypes(
        self,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        insecticide: phenotype_params.insecticide = None,
        dose: phenotype_params.dose = None,
        phenotype: phenotype_params.phenotype = None,
        region: Optional[base_params.region] = None,
        analysis: str = "arab",
        cohort_size: Optional[base_params.cohort_size] = None,
        min_cohort_size: Optional[base_params.min_cohort_size] = None,
        max_cohort_size: Optional[base_params.max_cohort_size] = None,
    ) -> xr.Dataset:
        """
        Load phenotypic data combined with genetic variant data.
        """

        # 1. Get phenotypic DataFrame
        df_phenotypes = self.phenotype_data(
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            insecticide=insecticide,
            dose=dose,
            phenotype=phenotype,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
        )
        if df_phenotypes.empty:
            warnings.warn(
                "No phenotype data found for the specified parameters. Cannot merge genetic data."
            )
            return self._create_phenotype_dataset(
                df_phenotypes=df_phenotypes, variant_data=None
            )

        # 2. Optionally load variant data if region is specified
        variant_data = None
        if region is not None:
            sample_ids = df_phenotypes["sample_id"].unique().tolist()
            sample_query_for_genetics = f"sample_id in {sample_ids}"
            snp_error_message = None
            hap_error_message = None

            sample_sets_norm = self._prep_sample_sets_param(sample_sets=sample_sets)
            try:
                if hasattr(self, "snp_calls") and callable(self.snp_calls):
                    variant_data = self.snp_calls(
                        region=region,
                        sample_sets=sample_sets_norm,
                        sample_query=sample_query_for_genetics,
                    )
                    if (
                        variant_data is None
                        or variant_data.sizes.get("variants", 0) == 0
                    ):
                        variant_data = None
                        snp_error_message = "snp_calls returned no data"
                else:
                    snp_error_message = "snp_calls method not available"

            except Exception as e:
                snp_error_message = f"snp_calls failed with Exception: {e}"
                variant_data = None

            if variant_data is None:
                try:
                    if hasattr(self, "haplotypes") and callable(self.haplotypes):
                        hap_kwargs = {
                            "region": region,
                            "sample_sets": sample_sets_norm,
                            "sample_query": sample_query_for_genetics,
                        }
                        if analysis is not None:
                            hap_kwargs["analysis"] = analysis

                        variant_data = self.haplotypes(**hap_kwargs)

                        if (
                            variant_data is None
                            or variant_data.sizes.get("variants", 0) == 0
                        ):
                            variant_data = None
                            hap_error_message = "haplotypes returned no data"
                    else:
                        hap_error_message = "haplotypes method not available"

                except Exception as e:
                    hap_error_message = f"haplotypes failed with Exception: {e}"
                    variant_data = None

            # If both failed, issue a warning
            if variant_data is None:
                warnings.warn(
                    f"Could not load genetic data. snp_calls status: [{snp_error_message or 'Not attempted/Available'}]. haplotypes status: [{hap_error_message or 'Not attempted/Available'}]"
                )

        # 3. Merge into an xarray Dataset
        ds = self._create_phenotype_dataset(df_phenotypes, variant_data)
        return ds

    def phenotype_sample_sets(self) -> List[str]:
        """
        Get list of sample sets that have phenotypic data available.
        """
        all_sample_sets = self.sample_sets
        phenotype_sample_sets = []
        base_phenotype_path = f"{self._url}/v3.2/phenotypes/all"
        for sample_set in all_sample_sets:
            try:
                phenotype_path = f"{base_phenotype_path}/{sample_set}/phenotypes.csv"
                if self._fs.exists(phenotype_path):
                    phenotype_sample_sets.append(sample_set)
                if self._fs.exists(phenotype_path):
                    phenotype_sample_sets.append(sample_set)
            except Exception:
                continue
        return phenotype_sample_sets
