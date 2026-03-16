from typing import Mapping, Optional, Tuple

import allel  # type: ignore
import numpy as np
import pandas as pd
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
from numpydoc_decorator import doc  # type: ignore

from ..util import CacheMiss, _check_types, _jackknife_ci
from . import base_params, plotly_params
from .snp_data import AnophelesSnpData


class AnophelesDiversityAnalysis(
    AnophelesSnpData,
):
    def __init__(
        self,
        **kwargs,
    ):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

    def _block_jackknife_cohort_diversity_stats(
        self,
        *,
        cohort_label,
        ac,
        n_jack, 
        confidence_level
    ):
        debug = self._log.debug

        debug("set up for diversity calculations")
        n_sites = ac.shape[0]
        ac = allel.AlleleCountsArray(ac)
        n = ac.sum(axis=1).max()  # number of chromosomes sampled
        n_sites = min(n_sites, ac.shape[0])  # number of sites
        block_length = n_sites // n_jack  # number of sites in each block
        n_sites_j = n_sites - block_length  # number of sites in each jackknife resample

        debug("compute scaling constants")
        a1 = np.sum(1 / np.arange(1, n))
        a2 = np.sum(1 / (np.arange(1, n) ** 2))
        b1 = (n + 1) / (3 * (n - 1))
        b2 = 2 * (n**2 + n + 3) / (9 * n * (n - 1))
        c1 = b1 - (1 / a1)
        c2 = b2 - ((n + 2) / (a1 * n)) + (a2 / (a1**2))
        e1 = c1 / a1
        e2 = c2 / (a1**2 + a2)

        debug(
            "compute some intermediates ahead of time, to minimise computation during jackknife resampling"
        )
        mpd_data = allel.mean_pairwise_difference(ac, fill=0)
        # N.B., here we compute the number of segregating sites as the number
        # of alleles minus 1. This follows the sgkit and tskit implementations,
        # and is different from scikit-allel.
        seg_data = ac.allelism() - 1

        debug("compute estimates from all data")
        theta_pi_abs_data = np.sum(mpd_data)
        theta_pi_data = theta_pi_abs_data / n_sites
        S_data = np.sum(seg_data)
        theta_w_abs_data = S_data / a1
        theta_w_data = theta_w_abs_data / n_sites
        d_data = theta_pi_abs_data - theta_w_abs_data
        d_stdev_data = np.sqrt((e1 * S_data) + (e2 * S_data * (S_data - 1)))
        tajima_d_data = d_data / d_stdev_data

        debug("set up for jackknife resampling")
        jack_theta_pi = []
        jack_theta_w = []
        jack_tajima_d = []

        debug("begin jackknife resampling")
        for i in range(n_jack):
            # locate block to delete
            block_start = i * block_length
            block_stop = block_start + block_length
            loc_j = np.ones(n_sites, dtype=bool)
            loc_j[block_start:block_stop] = False
            assert np.count_nonzero(loc_j) == n_sites_j

            # resample data and compute statistics

            # theta_pi
            mpd_j = mpd_data[loc_j]
            theta_pi_abs_j = np.sum(mpd_j)
            theta_pi_j = theta_pi_abs_j / n_sites_j
            jack_theta_pi.append(theta_pi_j)

            # theta_w
            seg_j = seg_data[loc_j]
            S_j = np.sum(seg_j)
            theta_w_abs_j = S_j / a1
            theta_w_j = theta_w_abs_j / n_sites_j
            jack_theta_w.append(theta_w_j)

            # tajima_d
            d_j = theta_pi_abs_j - theta_w_abs_j
            d_stdev_j = np.sqrt((e1 * S_j) + (e2 * S_j * (S_j - 1)))
            tajima_d_j = d_j / d_stdev_j
            jack_tajima_d.append(tajima_d_j)

        # calculate jackknife stats
        (
            theta_pi_estimate,
            theta_pi_bias,
            theta_pi_std_err,
            theta_pi_ci_err,
            theta_pi_ci_low,
            theta_pi_ci_upp,
        ) = _jackknife_ci(
            stat_data=theta_pi_data,
            jack_stat=jack_theta_pi,
            confidence_level=confidence_level,
        )
        (
            theta_w_estimate,
            theta_w_bias,
            theta_w_std_err,
            theta_w_ci_err,
            theta_w_ci_low,
            theta_w_ci_upp,
        ) = _jackknife_ci(
            stat_data=theta_w_data,
            jack_stat=jack_theta_w,
            confidence_level=confidence_level,
        )
        (
            tajima_d_estimate,
            tajima_d_bias,
            tajima_d_std_err,
            tajima_d_ci_err,
            tajima_d_ci_low,
            tajima_d_ci_upp,
        ) = _jackknife_ci(
            stat_data=tajima_d_data,
            jack_stat=jack_tajima_d,
            confidence_level=confidence_level,
        )

        return dict(
            cohort=cohort_label,
            theta_pi=theta_pi_data,
            theta_pi_estimate=theta_pi_estimate,
            theta_pi_bias=theta_pi_bias,
            theta_pi_std_err=theta_pi_std_err,
            theta_pi_ci_err=theta_pi_ci_err,
            theta_pi_ci_low=theta_pi_ci_low,
            theta_pi_ci_upp=theta_pi_ci_upp,
            theta_w=theta_w_data,
            theta_w_estimate=theta_w_estimate,
            theta_w_bias=theta_w_bias,
            theta_w_std_err=theta_w_std_err,
            theta_w_ci_err=theta_w_ci_err,
            theta_w_ci_low=theta_w_ci_low,
            theta_w_ci_upp=theta_w_ci_upp,
            tajima_d=tajima_d_data,
            tajima_d_estimate=tajima_d_estimate,
            tajima_d_bias=tajima_d_bias,
            tajima_d_std_err=tajima_d_std_err,
            tajima_d_ci_err=tajima_d_ci_err,
            tajima_d_ci_low=tajima_d_ci_low,
            tajima_d_ci_upp=tajima_d_ci_upp,
        )

    @_check_types
    @doc(
        summary="""
            Compute genetic diversity summary statistics for a cohort of
            individuals.
        """,
        returns="""
            A pandas series with summary statistics (theta pi, Watterson's theta and Tajima's D)
            and their estimate, bias, standard error, confidence interval error, confidence interval lower value,
            and confidence interval upper value. The series also contains the cohort under study, its taxon, its year
            of collection, its month of collection, its country of collection, the ISO code of its first administrative
            level of collection, the name of its first administrative level of collection, the name of its second administrative
            level of collection, the longitude of its location of collection, and the latitude of its location of collection.
        """,
    )
    def cohort_diversity_stats(
        self,
        cohort: base_params.cohort,
        cohort_size: base_params.cohort_size,
        region: base_params.regions,
        min_cohort_size: Optional[base_params.min_cohort_size] = None,
        max_cohort_size: Optional[base_params.max_cohort_size] = None,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        site_class: Optional[base_params.site_class] = None,
        sample_sets: Optional[base_params.sample_sets] = None,
        random_seed: base_params.random_seed = 42,
        n_jack: base_params.n_jack = 200,
        confidence_level: base_params.confidence_level = 0.95,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ) -> pd.Series:
        debug = self._log.debug

        # Change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data.
        name = "cohort_diversity_stats_v1"

        debug("process cohort parameter")
        cohort_query = None
        if isinstance(cohort, str):
            # assume it is one of the predefined cohorts
            cohort_label = cohort
            df_samples = self.sample_metadata(sample_sets=sample_sets)
            cohort_cols = [c for c in df_samples.columns if c.startswith("cohort_")]
            for c in cohort_cols:
                if cohort in set(df_samples[c]):
                    cohort_query = f"{c} == '{cohort}'"
                    break
            if cohort_query is None:
                raise ValueError(f"unknown cohort: {cohort}")

        elif isinstance(cohort, (list, tuple)) and len(cohort) == 2:
            cohort_label, cohort_query = cohort

        else:
            raise TypeError(f"invalid cohort parameter: {cohort!r}")

        params = dict(
            cohort_label=cohort_label,
            cohort_query=cohort_query,
            cohort_size=cohort_size,
            region=region,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            site_mask=site_mask,
            site_class=site_class,
            sample_sets=sample_sets,
            random_seed=random_seed,
            n_jack=n_jack,
            confidence_level=confidence_level,
            chunks=chunks,
            inline_array=inline_array,
        )

        # Try to retrieve results from the cache.
        try:
            results = self.results_cache_get(name=name, params=params)
            stats = {
                key: (
                    value.item()
                    if isinstance(value, np.ndarray) and value.shape == ()
                    else value
                )
                for key, value in results.items()
            }

        except CacheMiss:
            debug("access allele counts")
            ac = self.snp_allele_counts(
                region=region,
                site_mask=site_mask,
                site_class=site_class,
                sample_query=cohort_query,
                sample_sets=sample_sets,
                cohort_size=cohort_size,
                min_cohort_size=min_cohort_size,
                max_cohort_size=max_cohort_size,
                random_seed=random_seed,
                chunks=chunks,
                inline_array=inline_array,
            )

            debug("compute diversity stats")
            stats = self._block_jackknife_cohort_diversity_stats(
                cohort_label=cohort_label,
                ac=ac,
                n_jack=n_jack,
                confidence_level=confidence_level,
            )

            cache_results = {key: np.asarray(value) for key, value in stats.items()}
            self.results_cache_set(name=name, params=params, results=cache_results)

        debug("compute some extra cohort variables")
        df_samples = self.sample_metadata(
            sample_sets=sample_sets, sample_query=cohort_query
        )
        extra_fields = [
            ("taxon", "unique"),
            ("year", "unique"),
            ("month", "unique"),
            ("country", "unique"),
            ("admin1_iso", "unique"),
            ("admin1_name", "unique"),
            ("admin2_name", "unique"),
            ("longitude", "mean"),
            ("latitude", "mean"),
        ]
        for field, agg in extra_fields:
            if agg == "unique":
                vals = df_samples[field].dropna().sort_values().unique()
                if len(vals) == 0:
                    val = np.nan
                elif len(vals) == 1:
                    val = vals[0]
                else:
                    val = vals.tolist()
            elif agg == "mean":
                vals = df_samples[field].dropna()
                if len(vals) == 0:
                    val = np.nan
                else:
                    val = np.mean(vals)
            else:
                val = np.nan
            stats[field] = val

        return pd.Series(stats)

    @_check_types
    @doc(
        summary="""
            Compute genetic diversity summary statistics for multiple cohorts.
        """,
        returns="""
            A DataFrame where each row provides summary statistics and their
            confidence intervals for a single cohort. The columns are
            the value, the estimate, the bias, the standard error,
            the confidence interval error, the confidence interval lower value,
            the confidence interval upper value for each summary statistics (theta pi, Watterson's theta and Tajima's D),
            the taxon of the cohort, its year
            of collection, its month of collection, its country of collection, the ISO code of its first administrative
            level of collection, the name of its first administrative level of collection, the name of its second administrative
            level of collection, the longitude of its location of collection, and the latitude of its location of collection.
        """,
    )
    def diversity_stats(
        self,
        cohorts: base_params.cohorts,
        cohort_size: base_params.cohort_size,
        region: base_params.regions,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        site_class: Optional[base_params.site_class] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        sample_sets: Optional[base_params.sample_sets] = None,
        random_seed: base_params.random_seed = 42,
        n_jack: base_params.n_jack = 200,
        confidence_level: base_params.confidence_level = 0.95,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ) -> pd.DataFrame:
        # Normalise cohorts parameter.
        cohort_queries = self._setup_cohort_queries(
            cohorts=cohorts,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            cohort_size=cohort_size,
            min_cohort_size=None,
        )

        # Compute diversity stats for cohorts.
        all_stats = []
        for cohort_label, cohort_query in cohort_queries.items():
            stats = self.cohort_diversity_stats(
                cohort=(cohort_label, cohort_query),
                cohort_size=cohort_size,
                region=region,
                site_mask=site_mask,
                site_class=site_class,
                sample_sets=sample_sets,
                random_seed=random_seed,
                n_jack=n_jack,
                confidence_level=confidence_level,
                chunks=chunks,
                inline_array=inline_array,
            )
            all_stats.append(stats)
        df_stats = pd.DataFrame(all_stats)

        return df_stats

    @_check_types
    @doc(
        summary="Plot diversity summary statistics for multiple cohorts.",
        parameters=dict(
            df_stats="Output from `diversity_stats()`.",
            bar_plot_height="Height of bar plots in pixels (px).",
            bar_width="Width per bar in pixels (px).",
            scatter_plot_height="Height of scatter plot in pixels (px).",
            scatter_plot_width="Width of scatter plot in pixels (px).",
            plot_kwargs="Extra plotting parameters.",
        ),
    )
    def plot_diversity_stats(
        self,
        df_stats: pd.DataFrame,
        color: plotly_params.color = None,
        bar_plot_height: int = 450,
        bar_width: int = 30,
        scatter_plot_height: int = 500,
        scatter_plot_width: int = 500,
        template: plotly_params.template = "plotly_white",
        color_discrete_sequence: plotly_params.color_discrete_sequence = None,
        color_discrete_map: plotly_params.color_discrete_map = None,
        category_orders: plotly_params.category_order = None,
        plot_kwargs: Optional[Mapping] = None,
        show: plotly_params.show = True,
        renderer: plotly_params.renderer = None,
    ) -> Optional[Tuple[go.Figure, ...]]:
        # Handle color.
        (
            color_prepped,
            color_discrete_map_prepped,
            category_orders_prepped,
        ) = self._setup_sample_colors_plotly(
            data=df_stats,
            color=color,
            color_discrete_map=color_discrete_map,
            color_discrete_sequence=color_discrete_sequence,
            category_orders=category_orders,
        )
        del color
        del color_discrete_map
        del color_discrete_sequence
        del category_orders

        # Set up common plotting parameters.
        default_plot_kwargs = dict(
            hover_name="cohort",
            hover_data=[
                "taxon",
                "country",
                "admin1_iso",
                "admin1_name",
                "admin2_name",
                "longitude",
                "latitude",
                "year",
                "month",
            ],
            labels={
                "theta_pi_estimate": "θ<sub>π</sub>",
                "theta_w_estimate": "θ<sub>𝑤</sub>",
                "tajima_d_estimate": "𝐷",
                "cohort": "Cohort",
                "taxon": "Taxon",
                "country": "Country",
            },
            color=color_prepped,
            color_discrete_map=color_discrete_map_prepped,
            category_orders=category_orders_prepped,
            template=template,
        )

        # Finalise parameters.
        if plot_kwargs is None:
            plot_kwargs = dict()
        default_plot_kwargs.update(plot_kwargs)
        plot_kwargs = default_plot_kwargs
        bar_plot_width = 300 + bar_width * len(df_stats)

        # Nucleotide diversity bar plot.
        fig1 = px.bar(
            data_frame=df_stats,
            x="cohort",
            y="theta_pi_estimate",
            error_y="theta_pi_ci_err",
            title="Nucleotide diversity",
            height=bar_plot_height,
            width=bar_plot_width,
            **plot_kwargs,
        )

        # Watterson's estimator bar plot.
        fig2 = px.bar(
            data_frame=df_stats,
            x="cohort",
            y="theta_w_estimate",
            error_y="theta_w_ci_err",
            title="Watterson's estimator",
            height=bar_plot_height,
            width=bar_plot_width,
            **plot_kwargs,
        )

        # Tajima's D bar plot.
        fig3 = px.bar(
            data_frame=df_stats,
            x="cohort",
            y="tajima_d_estimate",
            error_y="tajima_d_ci_err",
            title="Tajima's D",
            height=bar_plot_height,
            width=bar_plot_width,
            **plot_kwargs,
        )

        # Scatter plot comparing diversity estimators.
        fig4 = px.scatter(
            data_frame=df_stats,
            x="theta_pi_estimate",
            y="theta_w_estimate",
            error_x="theta_pi_ci_err",
            error_y="theta_w_ci_err",
            title="Diversity estimators",
            width=scatter_plot_width,
            height=scatter_plot_height,
            **plot_kwargs,
        )

        if show:  # pragma: no cover
            fig1.show(renderer=renderer)
            fig2.show(renderer=renderer)
            fig3.show(renderer=renderer)
            fig4.show(renderer=renderer)
            return None
        else:
            return (fig1, fig2, fig3, fig4)
