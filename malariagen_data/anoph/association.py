from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr
from scipy.stats import pointbiserialr  # type: ignore

from ..util import _check_types
from . import base_params


class AnophelesAssociationPlotting:
    """Helpers for creating and plotting association test results."""

    if TYPE_CHECKING:
        phenotypes_with_snps: Callable[..., xr.Dataset]

    @_check_types
    def association_results(
        self,
        data: xr.Dataset,
        *,
        genotype_col: str = "call_genotype",
        phenotype_col: str = "phenotype_binary",
        contig_col: str = "variant_contig",
        position_col: str = "variant_position",
    ) -> pd.DataFrame:
        """Compute per-variant association p-values from SNP/phenotype data."""

        required = (genotype_col, phenotype_col, contig_col, position_col)
        missing = [v for v in required if v not in data]
        if missing:
            raise ValueError(f"Missing required variables in dataset: {missing}")

        genotype = np.asarray(data[genotype_col].values)
        if genotype.ndim == 3:
            dosage = np.where(genotype < 0, np.nan, genotype).sum(axis=2)
        elif genotype.ndim == 2:
            dosage = np.where(genotype < 0, np.nan, genotype).astype(float)
        else:
            raise ValueError(
                f"{genotype_col!r} must have 2 or 3 dimensions, found {genotype.ndim}"
            )

        phenotype = np.asarray(data[phenotype_col].values, dtype=float)
        if phenotype.ndim != 1:
            raise ValueError(f"{phenotype_col!r} must be one-dimensional")
        if dosage.shape[1] != phenotype.shape[0]:
            raise ValueError(
                "Genotype and phenotype sample dimensions do not align: "
                f"{dosage.shape[1]} vs {phenotype.shape[0]}"
            )

        contig_values = np.asarray(data[contig_col].values)
        if np.issubdtype(contig_values.dtype, np.integer) and "contigs" in data.coords:
            contig_lookup = np.asarray(data.coords["contigs"].values, dtype="U")
            contigs = np.asarray(
                [
                    contig_lookup[c] if 0 <= c < contig_lookup.shape[0] else str(c)
                    for c in contig_values
                ],
                dtype="U",
            )
        else:
            contigs = contig_values.astype("U")

        positions = np.asarray(data[position_col].values, dtype=int)

        n_variants = dosage.shape[0]
        pvalues = np.full(n_variants, np.nan, dtype=float)
        n_obs = np.zeros(n_variants, dtype=int)

        phenotype_valid = np.isfinite(phenotype)
        for i in range(n_variants):
            x = dosage[i]
            mask = np.isfinite(x) & phenotype_valid
            n_obs[i] = int(mask.sum())
            if n_obs[i] < 3:
                continue

            x_valid = x[mask]
            y_valid = phenotype[mask]
            if np.unique(x_valid).size < 2 or np.unique(y_valid).size < 2:
                continue

            _, pvalue = pointbiserialr(y_valid, x_valid)
            pvalues[i] = float(pvalue)

        return pd.DataFrame(
            {
                "contig": contigs,
                "position": positions,
                "pvalue": pvalues,
                "n_obs": n_obs,
            }
        )

    @_check_types
    def snp_phenotype_association(
        self,
        *,
        region: base_params.region,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        cohort_size: Optional[base_params.cohort_size] = None,
        min_cohort_size: Optional[base_params.min_cohort_size] = None,
        max_cohort_size: Optional[base_params.max_cohort_size] = None,
    ) -> pd.DataFrame:
        """Generate association results from phenotype/SNP data for plotting."""

        ds = self.phenotypes_with_snps(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
        )

        if not isinstance(ds, xr.Dataset) or ds.sizes.get("variants", 0) == 0:
            raise ValueError(
                "No variant data available to compute association results."
            )

        return self.association_results(ds)

    @_check_types
    def _association_results_to_dataframe(
        self,
        data: Union[pd.DataFrame, xr.Dataset],
        required_columns: Sequence[str],
    ) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, xr.Dataset):
            missing_vars = [c for c in required_columns if c not in data]
            if missing_vars:
                raise ValueError(
                    f"Missing required variables in xarray dataset: {missing_vars}"
                )
            df = data[list(required_columns)].to_dataframe().reset_index()
        else:
            raise TypeError("data must be a pandas DataFrame or xarray Dataset")

        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

        return df

    @_check_types
    def plot_manhattan(
        self,
        data: Union[pd.DataFrame, xr.Dataset],
        *,
        contig_col: str = "contig",
        position_col: str = "position",
        pvalue_col: str = "pvalue",
        contig_order: Optional[Sequence[str]] = None,
        contig_spacing: int = 1_000_000,
        pvalue_threshold: Optional[float] = 5e-8,
        width: int = 1000,
        height: int = 500,
        show: bool = True,
        renderer: Optional[str] = None,
        **kwargs: Any,
    ) -> go.Figure:
        df = self._association_results_to_dataframe(
            data,
            required_columns=(contig_col, position_col, pvalue_col),
        )
        df = df[[contig_col, position_col, pvalue_col]].copy()
        df = df.dropna(subset=[contig_col, position_col, pvalue_col])
        df = df[(df[pvalue_col] > 0) & (df[pvalue_col] <= 1)]
        if df.empty:
            raise ValueError("No valid p-values found for Manhattan plot.")

        if contig_order is None:
            contigs = list(df[contig_col].astype(str).dropna().unique())
        else:
            observed_contigs = set(df[contig_col].astype(str).dropna().unique())
            contigs = [c for c in contig_order if c in observed_contigs]
            missing_from_order = observed_contigs - set(contigs)
            contigs.extend(sorted(missing_from_order))

        contig_offsets = {}
        running_offset = 0.0
        tickvals = []
        ticktext = []

        for contig in contigs:
            df_contig = df[df[contig_col].astype(str) == contig]
            min_pos = float(df_contig[position_col].min())
            max_pos = float(df_contig[position_col].max())
            contig_offsets[contig] = running_offset - min_pos
            tickvals.append(running_offset + (max_pos - min_pos) / 2)
            ticktext.append(contig)
            running_offset += (max_pos - min_pos) + contig_spacing

        df["_contig"] = df[contig_col].astype(str)
        df["_x"] = df[position_col].astype(float) + df["_contig"].map(contig_offsets)
        df["_minus_log10_p"] = -np.log10(df[pvalue_col].astype(float))

        fig = px.scatter(
            df,
            x="_x",
            y="_minus_log10_p",
            color="_contig",
            labels={
                "_x": "Genomic position",
                "_minus_log10_p": "-log10(p-value)",
                "_contig": contig_col,
            },
            width=width,
            height=height,
            template="simple_white",
            **kwargs,
        )
        fig.update_layout(
            xaxis=dict(tickmode="array", tickvals=tickvals, ticktext=ticktext),
            legend_title_text=contig_col,
        )

        if pvalue_threshold is not None and 0 < pvalue_threshold <= 1:
            fig.add_hline(
                y=-np.log10(pvalue_threshold),
                line_dash="dash",
                line_color="red",
            )

        if show:  # pragma: no cover
            fig.show(renderer=renderer)
        return fig

    @_check_types
    def plot_qq(
        self,
        data: Union[pd.DataFrame, xr.Dataset],
        *,
        pvalue_col: str = "pvalue",
        width: int = 600,
        height: int = 600,
        show: bool = True,
        renderer: Optional[str] = None,
        **kwargs: Any,
    ) -> go.Figure:
        df = self._association_results_to_dataframe(
            data,
            required_columns=(pvalue_col,),
        )
        pvals_series = df[pvalue_col].dropna().astype(float)
        pvals_array = pvals_series[(pvals_series > 0) & (pvals_series <= 1)].to_numpy()
        if pvals_array.size == 0:
            raise ValueError("No valid p-values found for QQ plot.")

        pvals_array.sort()
        observed = -np.log10(pvals_array)
        expected = -np.log10(
            np.arange(1, pvals_array.size + 1) / (pvals_array.size + 1)
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=expected,
                y=observed,
                mode="markers",
                name="Observed",
            )
        )
        max_val = float(max(expected.max(), observed.max()))
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="Expected",
                line=dict(dash="dash", color="red"),
            )
        )
        fig.update_layout(
            template="simple_white",
            width=width,
            height=height,
            xaxis_title="Expected -log10(p-value)",
            yaxis_title="Observed -log10(p-value)",
            **kwargs,
        )

        if show:  # pragma: no cover
            fig.show(renderer=renderer)
        return fig
