import pandas as pd
import xarray as xr
from typing import Callable, Optional, Union, List, Dict, Any
import warnings
import fsspec


class PhenotypeDataMixin:
    """
    Mixin providing methods for accessing insecticide resistance phenotypic data.
    """

    _fs: fsspec.AbstractFileSystem
    _phenotype_gcs_path_template: str
    sample_metadata: Callable[..., pd.DataFrame]
    sample_sets: list[str]
    snp_calls: Callable[..., Any]
    haplotypes: Callable[..., Any]

    def _load_phenotype_data(
        self,
        sample_sets: list[str],
        insecticide: list[str] | None = None,
        dose: list[float] | None = None,
        phenotype: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load raw phenotypic data from GCS for given sample sets.
        """
        phenotype_dfs = []

        for sample_set in sample_sets:
            phenotype_path = self._phenotype_gcs_path_template.format(
                sample_set=sample_set
            )
            try:
                if not self._fs.exists(phenotype_path):
                    warnings.warn(f"Phenotype data file not found for {sample_set}")
                    continue

                with self._fs.open(phenotype_path, "r") as f:
                    df_pheno = pd.read_csv(f)

                df_pheno["sample_set"] = sample_set
                phenotype_dfs.append(df_pheno)

            except FileNotFoundError:
                warnings.warn(f"Phenotype data file not found for {sample_set}")
                continue
            except Exception as e:
                warnings.warn(
                    f"Unexpected error loading phenotype data for {sample_set}: {e}"
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
            df_combined = df_combined[df_combined["dose"].isin(dose)]
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
        df_merged = df_phenotypes.merge(
            df_sample_metadata,
            on="sample_id",
            how="left",
            suffixes=("", "_meta"),
        )

        duplicate_cols = [col for col in df_merged.columns if col.endswith("_meta")]
        for col in duplicate_cols:
            base_col = col.replace("_meta", "")
            if base_col in df_merged.columns:
                df_merged[base_col] = df_merged[base_col].fillna(df_merged[col])
            df_merged = df_merged.drop(columns=[col])

        return df_merged

    def _apply_phenotype_cohort_filtering(
        self,
        df: pd.DataFrame,
        cohort_size: Optional[int] = None,
        min_cohort_size: Optional[int] = None,
        max_cohort_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Apply cohort size filtering based on insecticide resistance experimental groups.
        """
        if all(
            param is None for param in [cohort_size, min_cohort_size, max_cohort_size]
        ):
            return df

        cohort_keys = ["insecticide", "dose", "location", "country"]
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
        Convert phenotype outcomes to binary format.
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
        binary_series = df["phenotype"].str.lower().map(phenotype_map)

        unmapped = df["phenotype"][binary_series.isna()]
        if not unmapped.empty:
            warnings.warn(f"Unmapped phenotype values found: {unmapped.unique()}")

        return binary_series

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

        data_vars = {
            "phenotype_binary": (["samples"], phenotype_binary.values),
            "insecticide": (["samples"], df_indexed["insecticide"].values),
            "dose": (["samples"], df_indexed["dose"].values),
            "phenotype": (["samples"], df_indexed["phenotype"].values),
        }

        # Optional variables
        for var in ["location", "country", "collection_date", "species"]:
            if var in df_indexed.columns:
                data_vars[var] = (["samples"], df_indexed[var].values)

        coords = {"samples": sample_ids}
        ds = xr.Dataset(data_vars, coords=coords)

        if variant_data is not None:
            common_samples = list(set(sample_ids) & set(variant_data.samples.values))
            if not common_samples:
                warnings.warn(
                    "No common samples found between phenotype and variant data"
                )
            else:
                ds = ds.sel(samples=common_samples)
                variant_data_subset = variant_data.sel(samples=common_samples)
                ds = xr.merge([ds, variant_data_subset])

        return ds

    def _validate_phenotype_parameters(
        self,
        insecticide: Optional[Union[str, List[str]]] = None,
        dose: Optional[Union[float, List[float]]] = None,
        phenotype: Optional[Union[str, List[str]]] = None,
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
        sample_sets: Optional[Union[str, List[str]]] = None,
        sample_query: Optional[str] = None,
        sample_query_options: Optional[Dict[str, Any]] = None,
        insecticide: Optional[Union[str, List[str]]] = None,
        dose: Optional[Union[float, List[float]]] = None,
        phenotype: Optional[Union[str, List[str]]] = None,
        cohort_size: Optional[int] = None,
        min_cohort_size: Optional[int] = None,
        max_cohort_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load phenotypic data from insecticide resistance bioassays.
        """
        # 1. Normalize sample_sets
        if sample_sets is None:
            sample_sets = self.sample_sets
        elif isinstance(sample_sets, str):
            sample_sets = [sample_sets]

        # 2. Validate parameters
        insecticide, dose, phenotype = self._validate_phenotype_parameters(
            insecticide, dose, phenotype
        )

        # 3. Load raw phenotype data
        df_phenotypes = self._load_phenotype_data(
            sample_sets=sample_sets,
            insecticide=[insecticide] if isinstance(insecticide, str) else insecticide,
            dose=[dose] if isinstance(dose, float) else dose,
            phenotype=[phenotype] if isinstance(phenotype, str) else phenotype,
        )

        # 4. Get metadata for those samples
        sample_ids_with_phenotypes = df_phenotypes["sample_id"].unique().tolist()
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
            df_merged = self.sample_metadata(
                sample_query=sample_query,
                sample_query_options=sample_query_options,
                df=df_merged,
            )

        # 7. Apply cohort filtering
        df_final = self._apply_phenotype_cohort_filtering(
            df_merged, cohort_size, min_cohort_size, max_cohort_size
        )

        return df_final

    def phenotype_binary(
        self,
        sample_sets: Optional[Union[str, List[str]]] = None,
        sample_query: Optional[str] = None,
        sample_query_options: Optional[Dict[str, Any]] = None,
        insecticide: Optional[Union[str, List[str]]] = None,
        dose: Optional[Union[float, List[float]]] = None,
        phenotype: Optional[Union[str, List[str]]] = None,
        cohort_size: Optional[int] = None,
        min_cohort_size: Optional[int] = None,
        max_cohort_size: Optional[int] = None,
    ) -> pd.Series:
        """
        Load phenotypic data as binary outcomes (1=alive, 0=dead).
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
        binary_series = self._create_phenotype_binary_series(df)
        binary_series.index = df["sample_id"].index
        return binary_series

    def phenotypes(
        self,
        sample_sets: Optional[Union[str, List[str]]] = None,
        sample_query: Optional[str] = None,
        sample_query_options: Optional[Dict[str, Any]] = None,
        insecticide: Optional[Union[str, List[str]]] = None,
        dose: Optional[Union[float, List[float]]] = None,
        phenotype: Optional[Union[str, List[str]]] = None,
        region: Optional[Union[str, List[str]]] = None,
        analysis: str = "arab",
        cohort_size: Optional[int] = None,
        min_cohort_size: Optional[int] = None,
        max_cohort_size: Optional[int] = None,
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

        print(f"snp_calls is callable? {callable(self.snp_calls)}")
        print(f"haplotypes is callable? {callable(self.haplotypes)}")
        # 2. Optionally load variant data if region is specified
        variant_data = None
        if region is not None:
            sample_ids = df_phenotypes["sample_id"].unique().tolist()
            try:
                variant_data = self.snp_calls(
                    region=region,
                    sample_sets=sample_sets,
                    sample_query=f"sample_id in {sample_ids}",
                )
            except Exception as e:
                try:
                    variant_data = self.haplotypes(
                        region=region,
                        sample_sets=sample_sets,
                        sample_query=f"sample_id in {sample_ids}",
                        analysis=analysis,
                    )
                except Exception:
                    warnings.warn(f"Could not load genetic data: {e}")

        # 3. Merge into an xarray Dataset
        ds = self._create_phenotype_dataset(df_phenotypes, variant_data)
        return ds

    def phenotype_sample_sets(self) -> List[str]:
        """
        Get list of sample sets that have phenotypic data available.
        """
        all_sample_sets = self.sample_sets
        phenotype_sample_sets = []
        for sample_set in all_sample_sets:
            try:
                phenotype_path = self._phenotype_gcs_path_template.format(
                    sample_set=sample_set
                )
                if self._fs.exists(phenotype_path):
                    phenotype_sample_sets.append(sample_set)
            except Exception:
                continue
        return phenotype_sample_sets
