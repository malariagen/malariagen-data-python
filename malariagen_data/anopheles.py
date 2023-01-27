import json
import sys
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from textwrap import dedent

import allel
import dask.array as da
import ipinfo
import numba
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from tqdm.auto import tqdm
from tqdm.dask import TqdmCallback

try:
    # noinspection PyPackageRequirements
    from google import colab
except ImportError:
    colab = None

from . import veff
from .util import (
    DIM_ALLELE,
    DIM_PLOIDY,
    DIM_SAMPLE,
    DIM_VARIANT,
    CacheMiss,
    LoggingHelper,
    Region,
    da_compress,
    da_from_zarr,
    dask_compress_dataset,
    hash_params,
    init_filesystem,
    init_zarr_store,
    jackknife_ci,
    jitter,
    locate_region,
    read_gff3,
    resolve_region,
    type_error,
    unpack_gff3_attributes,
    xarray_concat,
)

# width in px for bokeh genome plots - leave as None to allow stretching
DEFAULT_GENOME_PLOT_WIDTH = None
# see also https://docs.bokeh.org/en/latest/docs/user_guide/layout.html#sizing-modes
DEFAULT_GENOME_PLOT_SIZING_MODE = "stretch_width"
DEFAULT_GENES_TRACK_HEIGHT = 120  # height in px for bokeh genes track plots

AA_CHANGE_QUERY = (
    "effect in ['NON_SYNONYMOUS_CODING', 'START_LOST', 'STOP_LOST', 'STOP_GAINED']"
)


class AnophelesDataResource(ABC):

    # TODO: parent class docstring

    def __init__(
        self,
        url,
        config_path,
        cohorts_analysis=None,
        site_filters_analysis=None,
        bokeh_output_notebook=True,
        results_cache=None,
        log=sys.stdout,
        debug=False,
        show_progress=True,
        check_location=True,
        pre=False,
        **kwargs,  # used by simplecache, init_filesystem(url, **kwargs)
    ):

        self._url = url
        self._pre = pre
        self._site_filters_analysis = site_filters_analysis
        self._debug = debug
        self._show_progress = show_progress

        # set up logging
        self._log = LoggingHelper(name=__name__, out=log, debug=debug)

        # set up filesystem
        self._fs, self._base_path = init_filesystem(url, **kwargs)

        # set up caches
        self._cache_releases = None
        self._cache_sample_sets = dict()
        self._cache_sample_set_to_release = None
        self._cache_general_metadata = dict()
        self._cache_site_filters = dict()
        self._cache_snp_sites = None
        self._cache_snp_genotypes = dict()
        self._cache_genome = None
        self._cache_annotator = None
        self._cache_genome_features = dict()
        self._cache_geneset = dict()
        self._cache_sample_metadata = dict()
        self._cache_site_annotations = None
        self._cache_locate_site_class = dict()
        self._cache_cohort_metadata = dict()

        if results_cache is not None:
            results_cache = Path(results_cache).expanduser().resolve()
        self._results_cache = results_cache

        # get bokeh to output plots to the notebook - this is a common gotcha,
        # users forget to do this and wonder why bokeh plots don't show
        if bokeh_output_notebook:
            import bokeh.io as bkio

            bkio.output_notebook(hide_banner=True)

        # Occasionally, colab will allocate a VM outside the US, e.g., in
        # Europe or Asia. Because the MalariaGEN data GCS bucket is located
        # in the US, this is usually bad for performance, because of
        # increased latency and lower bandwidth. Add a check for this and
        # issue a warning if not in the US.
        client_details = None
        if check_location:
            try:
                client_details = ipinfo.getHandler().getDetails()
                if (
                    self._gcs_url in self._url
                    and colab
                    and client_details.country != "US"
                ):
                    warnings.warn(
                        dedent(
                            """
                        Your currently allocated Google Colab VM is not located in the US.
                        This usually means that data access will be substantially slower.
                        If possible, select "Runtime > Disconnect and delete runtime" from
                        the menu to request a new VM and try again.
                    """
                        )
                    )
            except OSError:
                pass
        self._client_details = client_details

        # load config
        path = f"{self._base_path}/{config_path}"
        with self._fs.open(path) as f:
            self._config = json.load(f)

        # set analysis versions

        if cohorts_analysis is None:
            self._cohorts_analysis = self._config.get("DEFAULT_COHORTS_ANALYSIS")
        else:
            self._cohorts_analysis = cohorts_analysis

        if site_filters_analysis is None:
            self._site_filters_analysis = self._config.get(
                "DEFAULT_SITE_FILTERS_ANALYSIS"
            )
        else:
            self._site_filters_analysis = site_filters_analysis

    # Start of @property

    @property
    def _client_location(self):
        details = self._client_details
        if details is not None:
            region = details.region
            country = details.country
            location = f"{region}, {country}"
            if colab:
                location += " (colab)"
            elif hasattr(details, "hostname"):
                hostname = details.hostname
                if hostname.endswith("googleusercontent.com"):
                    location += " (Google Cloud)"
        else:
            location = "unknown"
        return location

    @property
    def releases(self):
        """The releases for which data are available at the given storage
        location."""
        if self._cache_releases is None:
            if self._pre:
                # Here we discover which releases are available, by listing the storage
                # directory and examining the subdirectories. This may include "pre-releases"
                # where data may be incomplete.
                sub_dirs = [p.split("/")[-1] for p in self._fs.ls(self._base_path)]
                releases = tuple(
                    sorted(
                        [
                            self._path_to_release(d)
                            for d in sub_dirs
                            # FIXME: this matches v3 and v3.1, but also v3001.1
                            if d.startswith(f"v{self._major_version_int}")
                            and self._fs.exists(f"{self._base_path}/{d}/manifest.tsv")
                        ]
                    )
                )
                if len(releases) == 0:
                    raise ValueError("No releases found.")
                self._cache_releases = releases
            else:
                self._cache_releases = self._public_releases
        return self._cache_releases

    # Start of @property @abstractmethod

    @property
    @abstractmethod
    def contigs(self):
        raise NotImplementedError("Must override contigs")

    @property
    @abstractmethod
    def _major_version_int(self):
        raise NotImplementedError("Must override _major_version_int")

    @property
    @abstractmethod
    def _major_version_gcs_str(self):
        raise NotImplementedError("Must override _major_version_gcs_str")

    @property
    @abstractmethod
    def _genome_fasta_path(self):
        raise NotImplementedError("Must override _genome_fasta_path")

    @property
    @abstractmethod
    def _genome_fai_path(self):
        raise NotImplementedError("Must override _genome_fai_path")

    @property
    @abstractmethod
    def _genome_zarr_path(self):
        raise NotImplementedError("Must override _genome_zarr_path")

    @property
    @abstractmethod
    def _genome_ref_id(self):
        raise NotImplementedError("Must override _genome_ref_id")

    @property
    @abstractmethod
    def _genome_ref_name(self):
        raise NotImplementedError("Must override _genome_ref_name")

    @property
    @abstractmethod
    def _gcs_url(self):
        raise NotImplementedError("Must override _gcs_url")

    @property
    @abstractmethod
    def _pca_results_cache_name(self):
        raise NotImplementedError("Must override _pca_results_cache_name")

    @property
    @abstractmethod
    def _snp_allele_counts_results_cache_name(self):
        raise NotImplementedError("Must override _snp_allele_counts_results_cache_name")

    @property
    @abstractmethod
    def _fst_gwss_results_cache_name(self):
        raise NotImplementedError("Must override _fst_gwss_results_cache_name")

    @property
    @abstractmethod
    def _default_site_mask(self):
        raise NotImplementedError("Must override _default_site_mask")

    @property
    def _geneset_gff3_path(self):
        return self._config.get("GENESET_GFF3_PATH")

    @property
    def _public_releases(self):
        releases = self._config.get("PUBLIC_RELEASES")
        if releases is not None:
            releases = tuple(releases)
        return releases

    @property
    @abstractmethod
    def _site_annotations_zarr_path(self):
        raise NotImplementedError("Must override _site_annotations_zarr_path")

    # Start of @abstractmethod

    @abstractmethod
    def __repr__(self):
        # Not all children have species calls or cohorts data.
        raise NotImplementedError("Must override __repr__")

    @abstractmethod
    def _repr_html_(self):
        # Not all children have species calls or cohorts data.
        raise NotImplementedError("Must override _repr_html_")

    @abstractmethod
    def _site_mask_ids(self):
        # Not all children have the same site mask ids.
        raise NotImplementedError("Must override _site_mask_ids")

    @abstractmethod
    def _sample_metadata(self, sample_set):
        # Not all children have species calls or cohorts data.
        raise NotImplementedError("Must override _sample_metadata")

    @abstractmethod
    def _transcript_to_gene_name(self, transcript):
        # children may have different manual overrides.
        raise NotImplementedError("Must override _transcript_to_gene_name")

    @abstractmethod
    def _view_alignments_add_site_filters_tracks(
        self, *, contig, visibility_window, tracks
    ):
        # default implementation, do nothing
        raise NotImplementedError(
            "Must override _view_alignments_add_site_filters_tracks"
        )

    def _results_cache_add_analysis_params(self, params):
        # default implementation, can be overridden if additional analysis
        # params are used
        params["cohorts_analysis"] = self._cohorts_analysis
        params["site_filters_analysis"] = self._site_filters_analysis

    def results_cache_get(self, *, name, params):
        debug = self._log.debug
        if self._results_cache is None:
            raise CacheMiss
        params = params.copy()
        self._results_cache_add_analysis_params(params)
        cache_key, _ = hash_params(params)
        cache_path = self._results_cache / name / cache_key
        results_path = cache_path / "results.npz"
        if not results_path.exists():
            raise CacheMiss
        results = np.load(results_path)
        debug(f"loaded {name}/{cache_key}")
        return results

    def results_cache_set(self, *, name, params, results):
        debug = self._log.debug
        if self._results_cache is None:
            debug("no results cache has been configured, do nothing")
            return
        params = params.copy()
        self._results_cache_add_analysis_params(params)
        cache_key, params_json = hash_params(params)
        cache_path = self._results_cache / name / cache_key
        cache_path.mkdir(exist_ok=True, parents=True)
        params_path = cache_path / "params.json"
        results_path = cache_path / "results.npz"
        with params_path.open(mode="w") as f:
            f.write(params_json)
        np.savez_compressed(results_path, **results)
        debug(f"saved {name}/{cache_key}")

    def snp_allele_counts(
        self,
        region,
        sample_sets=None,
        sample_query=None,
        site_mask=None,
        site_class=None,
        cohort_size=None,
        random_seed=42,
    ):
        """Compute SNP allele counts. This returns the number of times each
        SNP allele was observed in the selected samples.

        Parameters
        ----------
        region : str or Region
            Contig name (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_class : str, optional
            Select sites belonging to one of the following classes: CDS_DEG_4,
            (4-fold degenerate coding sites), CDS_DEG_2_SIMPLE (2-fold simple
            degenerate coding sites), CDS_DEG_0 (non-degenerate coding sites),
            INTRON_SHORT (introns shorter than 100 bp), INTRON_LONG (introns
            longer than 200 bp), INTRON_SPLICE_5PRIME (intron within 2 bp of
            5' splice site), INTRON_SPLICE_3PRIME (intron within 2 bp of 3'
            splice site), UTR_5PRIME (5' untranslated region), UTR_3PRIME (3'
            untranslated region), INTERGENIC (intergenic, more than 10 kbp from
            a gene).
        cohort_size : int, optional
            If provided, randomly down-sample to the given cohort size before
            computing allele counts.
        random_seed : int, optional
            Random seed used for down-sampling.

        Returns
        -------
        ac : np.ndarray
            A numpy array of shape (n_variants, 4), where the first column has
            the reference allele (0) counts, the second column has the first
            alternate allele (1) counts, the third column has the second
            alternate allele (2) counts, and the fourth column has the third
            alternate allele (3) counts.

        Notes
        -----
        This computation may take some time to run, depending on your computing
        environment. Results of this computation will be cached and re-used if
        the `results_cache` parameter was set when instantiating the Ag3 class.

        """

        # change this name if you ever change the behaviour of this function,
        # to invalidate any previously cached data
        name = self._snp_allele_counts_results_cache_name

        # normalize params for consistent hash value
        if isinstance(sample_query, str):
            # resolve query to a list of integers for more cache hits
            df_samples = self.sample_metadata(sample_sets=sample_sets)
            loc_samples = df_samples.eval(sample_query).values
            sample_query = np.nonzero(loc_samples)[0].tolist()
        params = dict(
            region=self.resolve_region(region).to_dict(),
            sample_sets=self._prep_sample_sets_arg(sample_sets=sample_sets),
            sample_query=sample_query,
            site_mask=site_mask,
            site_class=site_class,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )

        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._snp_allele_counts(**params)
            self.results_cache_set(name=name, params=params, results=results)

        ac = results["ac"]
        return ac

    def snp_allele_frequencies_advanced(
        self,
        transcript,
        area_by,
        period_by,
        sample_sets=None,
        sample_query=None,
        min_cohort_size=10,
        drop_invariant=True,
        variant_query=None,
        site_mask=None,
        nobs_mode="called",  # or "fixed"
        ci_method="wilson",
    ):
        """Group samples by taxon, area (space) and period (time), then compute
        SNP allele counts and frequencies.

        Parameters
        ----------
        transcript : str
            Gene transcript ID, e.g., "AGAP004707-RD".
        area_by : str
            Column name in the sample metadata to use to group samples
            spatially. E.g., use "admin1_iso" or "admin1_name" to group by level
            1 administrative divisions, or use "admin2_name" to group by level 2
            administrative divisions.
        period_by : {"year", "quarter", "month"}
            Length of time to group samples temporally.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        min_cohort_size : int, optional
            Minimum cohort size. Any cohorts below this size are omitted.
        drop_invariant : bool, optional
            If True, variants with no alternate allele calls in any cohorts are
            dropped from the result.
        variant_query : str, optional
        site_mask : str, optional
            Site filters mask to apply.
        nobs_mode : {"called", "fixed"}
            Method for calculating the denominator when computing frequencies.
            If "called" then use the number of called alleles, i.e., number of
            samples with non-missing genotype calls multiplied by 2. If "fixed"
            then use the number of samples multiplied by 2.
        ci_method : {"normal", "agresti_coull", "beta", "wilson", "binom_test"}, optional
            Method to use for computing confidence intervals, passed through to
            `statsmodels.stats.proportion.proportion_confint`.

        Returns
        -------
        ds : xarray.Dataset
            The resulting dataset contains data has dimensions "cohorts" and
            "variants". Variables prefixed with "cohort" are 1-dimensional
            arrays with data about the cohorts, such as the area, period, taxon
            and cohort size. Variables prefixed with "variant" are
            1-dimensional arrays with data about the variants, such as the
            contig, position, reference and alternate alleles. Variables
            prefixed with "event" are 2-dimensional arrays with the allele
            counts and frequency calculations.

        """
        debug = self._log.debug

        debug("check parameters")
        self._check_param_min_cohort_size(min_cohort_size)

        debug("load sample metadata")
        df_samples = self.sample_metadata(
            sample_sets=sample_sets, sample_query=sample_query
        )

        debug("access SNP calls")
        ds_snps = self.snp_calls(
            region=transcript,
            sample_sets=sample_sets,
            sample_query=sample_query,
            site_mask=site_mask,
        )

        debug("access genotypes")
        gt = ds_snps["call_genotype"].data

        debug("prepare sample metadata for cohort grouping")
        df_samples = self._prep_samples_for_cohort_grouping(
            df_samples=df_samples,
            area_by=area_by,
            period_by=period_by,
        )

        debug("group samples to make cohorts")
        group_samples_by_cohort = df_samples.groupby(["taxon", "area", "period"])

        debug("build cohorts dataframe")
        df_cohorts = self._build_cohorts_from_sample_grouping(
            group_samples_by_cohort, min_cohort_size
        )

        debug("bring genotypes into memory")
        with self._dask_progress(desc="Load SNP genotypes"):
            gt = gt.compute()

        debug("set up variant variables")
        contigs = ds_snps.attrs["contigs"]
        variant_contig = np.repeat(
            [contigs[i] for i in ds_snps["variant_contig"].values], 3
        )
        variant_position = np.repeat(ds_snps["variant_position"].values, 3)
        alleles = ds_snps["variant_allele"].values
        variant_ref_allele = np.repeat(alleles[:, 0], 3)
        variant_alt_allele = alleles[:, 1:].flatten()
        variant_pass = dict()
        for site_mask in self._site_mask_ids():
            variant_pass[site_mask] = np.repeat(
                ds_snps[f"variant_filter_pass_{site_mask}"].values, 3
            )

        debug("setup main event variables")
        n_variants, n_cohorts = len(variant_position), len(df_cohorts)
        count = np.zeros((n_variants, n_cohorts), dtype=int)
        nobs = np.zeros((n_variants, n_cohorts), dtype=int)

        debug("build event count and nobs for each cohort")
        cohorts_iterator = self._progress(
            enumerate(df_cohorts.itertuples()),
            total=len(df_cohorts),
            desc="Compute SNP allele frequencies",
        )
        for cohort_index, cohort in cohorts_iterator:

            cohort_key = cohort.taxon, cohort.area, cohort.period
            sample_indices = group_samples_by_cohort.indices[cohort_key]

            cohort_ac, cohort_an = self._cohort_alt_allele_counts_melt(
                gt, sample_indices, max_allele=3
            )
            count[:, cohort_index] = cohort_ac

            if nobs_mode == "called":
                nobs[:, cohort_index] = cohort_an
            elif nobs_mode == "fixed":
                nobs[:, cohort_index] = cohort.size * 2
            else:
                raise ValueError(f"Bad nobs_mode: {nobs_mode!r}")

        debug("compute frequency")
        with np.errstate(divide="ignore", invalid="ignore"):
            # ignore division warnings
            frequency = count / nobs

        debug("compute maximum frequency over cohorts")
        with warnings.catch_warnings():
            # ignore "All-NaN slice encountered" warnings
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_af = np.nanmax(frequency, axis=1)

        debug("make dataframe of SNPs")
        df_variants_cols = {
            "contig": variant_contig,
            "position": variant_position,
            "ref_allele": variant_ref_allele.astype("U1"),
            "alt_allele": variant_alt_allele.astype("U1"),
            "max_af": max_af,
        }
        for site_mask in self._site_mask_ids():
            df_variants_cols[f"pass_{site_mask}"] = variant_pass[site_mask]
        df_variants = pd.DataFrame(df_variants_cols)

        debug("deal with SNP alleles not observed")
        if drop_invariant:
            loc_variant = max_af > 0
            df_variants = df_variants.loc[loc_variant].reset_index(drop=True)
            count = np.compress(loc_variant, count, axis=0)
            nobs = np.compress(loc_variant, nobs, axis=0)
            frequency = np.compress(loc_variant, frequency, axis=0)

        debug("set up variant effect annotator")
        ann = self._annotator()

        debug("add effects to the dataframe")
        ann.get_effects(
            transcript=transcript, variants=df_variants, progress=self._progress
        )

        debug("add variant labels")
        df_variants["label"] = self._pandas_apply(
            self._make_snp_label_effect,
            df_variants,
            columns=["contig", "position", "ref_allele", "alt_allele", "aa_change"],
        )

        debug("build the output dataset")
        ds_out = xr.Dataset()

        debug("cohort variables")
        for coh_col in df_cohorts.columns:
            ds_out[f"cohort_{coh_col}"] = "cohorts", df_cohorts[coh_col]

        debug("variant variables")
        for snp_col in df_variants.columns:
            ds_out[f"variant_{snp_col}"] = "variants", df_variants[snp_col]

        debug("event variables")
        ds_out["event_count"] = ("variants", "cohorts"), count
        ds_out["event_nobs"] = ("variants", "cohorts"), nobs
        ds_out["event_frequency"] = ("variants", "cohorts"), frequency

        debug("apply variant query")
        if variant_query is not None:
            loc_variants = df_variants.eval(variant_query).values
            ds_out = ds_out.isel(variants=loc_variants)

        debug("add confidence intervals")
        self._add_frequency_ci(ds_out, ci_method)

        debug("tidy up display by sorting variables")
        ds_out = ds_out[sorted(ds_out)]

        debug("add metadata")
        gene_name = self._transcript_to_gene_name(transcript)
        title = transcript
        if gene_name:
            title += f" ({gene_name})"
        title += " SNP frequencies"
        ds_out.attrs["title"] = title

        return ds_out

    # Start of @staticmethod @abstractmethod

    @staticmethod
    @abstractmethod
    def _setup_taxon_colors(plot_kwargs):
        # Subclasses have different taxon_color_map.
        raise NotImplementedError("Must override _setup_taxon_colors")

    # Start of @staticmethod

    @staticmethod
    def _bokeh_style_genome_xaxis(fig, contig):
        """Standard styling for X axis of genome plots."""
        import bokeh.models as bkmod

        fig.xaxis.axis_label = f"Contig {contig} position (bp)"
        fig.xaxis.ticker = bkmod.AdaptiveTicker(min_interval=1)
        fig.xaxis.minor_tick_line_color = None
        fig.xaxis[0].formatter = bkmod.NumeralTickFormatter(format="0,0")

    @staticmethod
    def _locate_cohorts(*, cohorts, df_samples):

        # build cohort dictionary where key=cohort_id, value=loc_coh
        coh_dict = {}

        if isinstance(cohorts, dict):
            # user has supplied a custom dictionary mapping cohort identifiers
            # to pandas queries

            for coh, query in cohorts.items():
                # locate samples
                loc_coh = df_samples.eval(query).values
                coh_dict[coh] = loc_coh

        if isinstance(cohorts, str):
            # user has supplied one of the predefined cohort sets

            # fix the string to match columns
            if not cohorts.startswith("cohort_"):
                cohorts = "cohort_" + cohorts

            # check the given cohort set exists
            if cohorts not in df_samples.columns:
                raise ValueError(f"{cohorts!r} is not a known cohort set")
            cohort_labels = df_samples[cohorts].unique()

            # remove the nans and sort
            cohort_labels = sorted([c for c in cohort_labels if isinstance(c, str)])
            for coh in cohort_labels:
                loc_coh = df_samples[cohorts] == coh
                coh_dict[coh] = loc_coh.values

        return coh_dict

    @staticmethod
    def _make_sample_period_month(row):
        year = row.year
        month = row.month
        if year > 0 and month > 0:
            return pd.Period(freq="M", year=year, month=month)
        else:
            return pd.NaT

    @staticmethod
    def _make_sample_period_quarter(row):
        year = row.year
        month = row.month
        if year > 0 and month > 0:
            return pd.Period(freq="Q", year=year, month=month)
        else:
            return pd.NaT

    @staticmethod
    def _make_sample_period_year(row):
        year = row.year
        if year > 0:
            return pd.Period(freq="A", year=year)
        else:
            return pd.NaT

    @staticmethod
    @numba.njit
    def _cohort_alt_allele_counts_melt_kernel(gt, indices, max_allele):

        n_variants = gt.shape[0]
        n_indices = indices.shape[0]
        ploidy = gt.shape[2]

        ac_alt_melt = np.zeros(n_variants * max_allele, dtype=np.int64)
        an = np.zeros(n_variants, dtype=np.int64)

        for i in range(n_variants):
            out_i_offset = (i * max_allele) - 1
            for j in range(n_indices):
                ix = indices[j]
                for k in range(ploidy):
                    allele = gt[i, ix, k]
                    if allele > 0:
                        out_i = out_i_offset + allele
                        ac_alt_melt[out_i] += 1
                        an[i] += 1
                    elif allele == 0:
                        an[i] += 1

        return ac_alt_melt, an

    @staticmethod
    def _make_snp_label(contig, position, ref_allele, alt_allele):
        return f"{contig}:{position:,} {ref_allele}>{alt_allele}"

    @staticmethod
    def _make_snp_label_effect(contig, position, ref_allele, alt_allele, aa_change):
        label = f"{contig}:{position:,} {ref_allele}>{alt_allele}"
        if isinstance(aa_change, str):
            label += f" ({aa_change})"
        return label

    @staticmethod
    def _make_snp_label_aa(aa_change, contig, position, ref_allele, alt_allele):
        label = f"{aa_change} ({contig}:{position:,} {ref_allele}>{alt_allele})"
        return label

    @staticmethod
    def _make_gene_cnv_label(gene_id, gene_name, cnv_type):
        label = gene_id
        if isinstance(gene_name, str):
            label += f" ({gene_name})"
        label += f" {cnv_type}"
        return label

    @staticmethod
    def _map_snp_to_aa_change_frq_ds(ds):

        # keep only variables that make sense for amino acid substitutions
        keep_vars = [
            "variant_contig",
            "variant_position",
            "variant_transcript",
            "variant_effect",
            "variant_impact",
            "variant_aa_pos",
            "variant_aa_change",
            "variant_ref_allele",
            "variant_ref_aa",
            "variant_alt_aa",
            "event_nobs",
        ]

        if ds.dims["variants"] == 1:

            # keep everything as-is, no need for aggregation
            ds_out = ds[keep_vars + ["variant_alt_allele", "event_count"]]

        else:

            # take the first value from all variants variables
            ds_out = ds[keep_vars].isel(variants=[0])

            # sum event count over variants
            count = ds["event_count"].values.sum(axis=0, keepdims=True)
            ds_out["event_count"] = ("variants", "cohorts"), count

            # collapse alt allele
            alt_allele = "{" + ",".join(ds["variant_alt_allele"].values) + "}"
            ds_out["variant_alt_allele"] = "variants", np.array(
                [alt_allele], dtype=object
            )

        return ds_out

    @staticmethod
    def _add_frequency_ci(ds, ci_method):
        from statsmodels.stats.proportion import proportion_confint

        if ci_method is not None:
            count = ds["event_count"].values
            nobs = ds["event_nobs"].values
            with np.errstate(divide="ignore", invalid="ignore"):
                frq_ci_low, frq_ci_upp = proportion_confint(
                    count=count, nobs=nobs, method=ci_method
                )
            ds["event_frequency_ci_low"] = ("variants", "cohorts"), frq_ci_low
            ds["event_frequency_ci_upp"] = ("variants", "cohorts"), frq_ci_upp

    @staticmethod
    def _build_cohorts_from_sample_grouping(group_samples_by_cohort, min_cohort_size):

        # build cohorts dataframe
        df_cohorts = group_samples_by_cohort.agg(
            size=("sample_id", len),
            lat_mean=("latitude", "mean"),
            lat_max=("latitude", "mean"),
            lat_min=("latitude", "mean"),
            lon_mean=("longitude", "mean"),
            lon_max=("longitude", "mean"),
            lon_min=("longitude", "mean"),
        )
        # reset index so that the index fields are included as columns
        df_cohorts = df_cohorts.reset_index()

        # add cohort helper variables
        cohort_period_start = df_cohorts["period"].apply(lambda v: v.start_time)
        cohort_period_end = df_cohorts["period"].apply(lambda v: v.end_time)
        df_cohorts["period_start"] = cohort_period_start
        df_cohorts["period_end"] = cohort_period_end
        # create a label that is similar to the cohort metadata,
        # although this won't be perfect
        df_cohorts["label"] = df_cohorts.apply(
            lambda v: f"{v.area}_{v.taxon[:4]}_{v.period}", axis="columns"
        )

        # apply minimum cohort size
        df_cohorts = df_cohorts.query(f"size >= {min_cohort_size}").reset_index(
            drop=True
        )

        return df_cohorts

    @staticmethod
    def _check_param_min_cohort_size(min_cohort_size):
        if not isinstance(min_cohort_size, int):
            raise TypeError(
                f"Type of parameter min_cohort_size must be int; found {type(min_cohort_size)}."
            )
        if min_cohort_size < 1:
            raise ValueError(
                f"Value of parameter min_cohort_size must be at least 1; found {min_cohort_size}."
            )

    @staticmethod
    def _pandas_apply(f, df, columns):
        """Optimised alternative to pandas apply."""
        df = df.reset_index(drop=True)
        iterator = zip(*[df[c].values for c in columns])
        ret = pd.Series((f(*vals) for vals in iterator))
        return ret

    @staticmethod
    def _roh_hmm_predict(
        *,
        windows,
        counts,
        phet_roh,
        phet_nonroh,
        transition,
        window_size,
        sample_id,
        contig,
    ):
        # conditional import, pomegranate takes a long time to install on
        # linux due to lack of prebuilt wheels on PyPI
        from allel.stats.misc import tabulate_state_blocks

        # this implementation is based on scikit-allel, but modified to use
        # moving window computation of het counts
        from allel.stats.roh import _hmm_derive_transition_matrix

        # FIXME: Unresolved references
        # noinspection PyUnresolvedReferences
        from pomegranate import HiddenMarkovModel, PoissonDistribution

        # het probabilities
        het_px = np.concatenate([(phet_roh,), phet_nonroh])

        # start probabilities (all equal)
        start_prob = np.repeat(1 / het_px.size, het_px.size)

        # transition between underlying states
        transition_mx = _hmm_derive_transition_matrix(transition, het_px.size)

        # emission probability distribution
        dists = [PoissonDistribution(x * window_size) for x in het_px]

        # set up model
        # noinspection PyArgumentList
        model = HiddenMarkovModel.from_matrix(
            transition_probabilities=transition_mx,
            distributions=dists,
            starts=start_prob,
        )

        # predict hidden states
        prediction = np.array(model.predict(counts[:, None]))

        # tabulate runs of homozygosity (state 0)
        # noinspection PyTypeChecker
        df_blocks = tabulate_state_blocks(prediction, states=list(range(len(het_px))))
        df_roh = df_blocks[(df_blocks["state"] == 0)].reset_index(drop=True)

        # adapt the dataframe for ROH
        df_roh["sample_id"] = sample_id
        df_roh["contig"] = contig
        df_roh["roh_start"] = df_roh["start_ridx"].apply(lambda y: windows[y, 0])
        df_roh["roh_stop"] = df_roh["stop_lidx"].apply(lambda y: windows[y, 1])
        df_roh["roh_length"] = df_roh["roh_stop"] - df_roh["roh_start"]
        df_roh.rename(columns={"is_marginal": "roh_is_marginal"}, inplace=True)

        return df_roh[
            [
                "sample_id",
                "contig",
                "roh_start",
                "roh_stop",
                "roh_length",
                "roh_is_marginal",
            ]
        ]

    # Start of undecorated functions

    # Note regarding release identifiers and storage paths. Within the
    # data storage, we have used path segments like "v3", "v3.1", "v3.2",
    # etc., to separate data from different releases. There is an inconsistency
    # in this convention, because the "v3" should have been "v3.0". To
    # make the API more consistent, we would like to use consistent release
    # identifiers like "3.0", "3.1", "3.2", etc., as parameter values and
    # when release identifiers are added to returned dataframes. In order to
    # achieve this, below we define two functions that allow mapping between
    # these consistent release identifiers, and the less consistent release
    # storage path segments.

    def _release_to_path(self, release):
        """Compatibility function, allows us to use release identifiers like "3.0"
        and "3.1" in the public API, and map these internally into storage path
        segments."""
        if release == f"{self._major_version_int}.0":
            # special case
            return self._major_version_gcs_str
        elif release.startswith(f"{self._major_version_int}."):
            return f"v{release}"
        else:
            raise ValueError(f"Invalid release: {release!r}")

    def _path_to_release(self, path):
        """Compatibility function, allows us to use release identifiers like "3.0"
        and "3.1" in the public API, and map these internally into storage path
        segments."""
        if path == self._major_version_gcs_str:
            return f"{self._major_version_int}.0"
        elif path.startswith(f"v{self._major_version_int}."):
            return path[1:]
        else:
            raise RuntimeError(f"Unexpected release path: {path!r}")

    def open_site_filters(self, mask):
        """Open site filters zarr.

        Parameters
        ----------
        mask : str
            Mask to use, e.g. "gamb_colu".

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        try:
            return self._cache_site_filters[mask]
        except KeyError:
            path = f"{self._base_path}/{self._major_version_gcs_str}/site_filters/{self._site_filters_analysis}/{mask}/"
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_site_filters[mask] = root
            return root

    def open_snp_sites(self):
        """Open SNP sites zarr.

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        if self._cache_snp_sites is None:
            path = f"{self._base_path}/{self._major_version_gcs_str}/snp_genotypes/all/sites/"
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_snp_sites = root
        return self._cache_snp_sites

    def _read_sample_sets(self, *, release):
        """Read the manifest of sample sets for a given release."""
        release_path = self._release_to_path(release)
        path = f"{self._base_path}/{release_path}/manifest.tsv"
        with self._fs.open(path) as f:
            df = pd.read_csv(f, sep="\t", na_values="")
        df["release"] = release
        return df

    def sample_sets(self, release=None):
        """Access a dataframe of sample sets.

        Parameters
        ----------
        release : str, optional
            Release identifier, e.g. give "3.0" to access the v3.0 data release.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of sample sets, one row per sample set.

        """

        if release is None:
            # retrieve sample sets from all available releases
            release = self.releases

        if isinstance(release, str):
            # retrieve sample sets for a single release

            if release not in self.releases:
                raise ValueError(f"Release not available: {release!r}")

            try:
                df = self._cache_sample_sets[release]

            except KeyError:
                df = self._read_sample_sets(release=release)
                self._cache_sample_sets[release] = df

        elif isinstance(release, (list, tuple)):

            # check no duplicates
            counter = Counter(release)
            for k, v in counter.items():
                if v > 1:
                    raise ValueError(f"Duplicate values: {k!r}.")

            # retrieve sample sets from multiple releases
            df = pd.concat(
                [self.sample_sets(release=r) for r in release],
                axis=0,
                ignore_index=True,
            )

        else:
            raise TypeError

        return df.copy()

    def _prep_sample_sets_arg(self, *, sample_sets):
        """Common handling for the `sample_sets` parameter. For convenience, we
        allow this to be a single sample set, or a list of sample sets, or a
        release identifier, or a list of release identifiers."""

        if sample_sets is None:
            # all available sample sets
            sample_sets = self.sample_sets()["sample_set"].tolist()

        elif isinstance(sample_sets, str):

            if sample_sets.startswith(f"{self._major_version_int}."):
                # convenience, can use a release identifier to denote all sample sets in a release
                sample_sets = self.sample_sets(release=sample_sets)[
                    "sample_set"
                ].tolist()

            else:
                # single sample set, normalise to always return a list
                sample_sets = [sample_sets]

        elif isinstance(sample_sets, (list, tuple)):
            # list or tuple of sample sets or releases
            prepped_sample_sets = []
            for s in sample_sets:

                # make a recursive call to handle the case where s is a release identifier
                sp = self._prep_sample_sets_arg(sample_sets=s)

                # make sure we end up with a flat list of sample sets
                if isinstance(sp, str):
                    prepped_sample_sets.append(sp)
                else:
                    prepped_sample_sets.extend(sp)
            sample_sets = prepped_sample_sets

        else:
            raise TypeError(
                f"Invalid type for sample_sets parameter; expected str, list or tuple; found: {sample_sets!r}"
            )

        # check all sample sets selected at most once
        counter = Counter(sample_sets)
        for k, v in counter.items():
            if v > 1:
                raise ValueError(
                    f"Bad value for sample_sets parameter, {k:!r} selected more than once."
                )

        return sample_sets

    def _progress(self, iterable, **kwargs):
        # progress doesn't mix well with debug logging
        disable = self._debug or not self._show_progress
        return tqdm(iterable, disable=disable, **kwargs)

    def _lookup_release(self, *, sample_set):
        """Find which release a sample set was included in."""

        if self._cache_sample_set_to_release is None:
            df_sample_sets = self.sample_sets().set_index("sample_set")
            self._cache_sample_set_to_release = df_sample_sets["release"].to_dict()

        try:
            return self._cache_sample_set_to_release[sample_set]
        except KeyError:
            raise ValueError(f"No release found for sample set {sample_set!r}")

    def _read_general_metadata(self, *, sample_set):
        """Read metadata for a single sample set."""
        try:
            df = self._cache_general_metadata[sample_set]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            release_path = self._release_to_path(release)
            path = f"{self._base_path}/{release_path}/metadata/general/{sample_set}/samples.meta.csv"
            dtype = {
                "sample_id": object,
                "partner_sample_id": object,
                "contributor": object,
                "country": object,
                "location": object,
                "year": "int64",
                "month": "int64",
                "latitude": "float64",
                "longitude": "float64",
                "sex_call": object,
            }
            with self._fs.open(path) as f:
                df = pd.read_csv(f, na_values="", dtype=dtype)

            # ensure all column names are lower case
            df.columns = [c.lower() for c in df.columns]

            # add a couple of columns for convenience
            df["sample_set"] = sample_set
            df["release"] = release

            self._cache_general_metadata[sample_set] = df
        return df.copy()

    def sample_metadata(
        self,
        sample_sets=None,
        sample_query=None,
    ):
        """Access sample metadata for one or more sample sets.

        Parameters
        ----------
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "country == 'Burkina Faso'".

        Returns
        -------
        df_samples : pandas.DataFrame
            A dataframe of sample metadata, one row per sample.

        """

        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)
        cache_key = tuple(sample_sets)

        try:

            df_samples = self._cache_sample_metadata[cache_key]

        except KeyError:

            # concatenate multiple sample sets
            dfs = []
            # there can be some delay here due to network latency, so show progress
            sample_sets_iterator = self._progress(
                sample_sets, desc="Load sample metadata"
            )
            for s in sample_sets_iterator:
                df = self._sample_metadata(sample_set=s)
                dfs.append(df)
            df_samples = pd.concat(dfs, axis=0, ignore_index=True)
            self._cache_sample_metadata[cache_key] = df_samples

        # for convenience, apply a query
        if sample_query is not None:
            df_samples = df_samples.query(sample_query).reset_index(drop=True)

        return df_samples.copy()

    def _site_filters(
        self,
        *,
        region,
        mask,
        field,
        inline_array,
        chunks,
    ):
        assert isinstance(region, Region)
        root = self.open_site_filters(mask=mask)
        z = root[f"{region.contig}/variants/{field}"]
        d = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        if region.start or region.end:
            root = self.open_snp_sites()
            pos = root[f"{region.contig}/variants/POS"][:]
            loc_region = locate_region(region, pos)
            d = d[loc_region]
        return d

    def site_filters(
        self,
        region,
        mask,
        field="filter_pass",
        inline_array=True,
        chunks="native",
    ):
        """Access SNP site filters.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Contig name (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        mask : str
            Mask to use, e.g. "gamb_colu"
        field : str, optional
            Array to access.
        inline_array : bool, optional
            Passed through to dask.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr
            chunks. Also, can be a target size, e.g., '200 MiB'.

        Returns
        -------
        d : dask.array.Array
            An array of boolean values identifying sites that pass the filters.

        """

        region = self.resolve_region(region)
        if isinstance(region, Region):
            region = [region]

        d = da.concatenate(
            [
                self._site_filters(
                    region=r,
                    mask=mask,
                    field=field,
                    inline_array=inline_array,
                    chunks=chunks,
                )
                for r in region
            ]
        )

        return d

    def _snp_sites(
        self,
        *,
        region,
        field,
        inline_array,
        chunks,
    ):
        assert isinstance(region, Region), type(region)
        root = self.open_snp_sites()
        z = root[f"{region.contig}/variants/{field}"]
        ret = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        if region.start or region.end:
            if field == "POS":
                pos = z[:]
            else:
                pos = root[f"{region.contig}/variants/POS"][:]
            loc_region = locate_region(region, pos)
            ret = ret[loc_region]
        return ret

    def snp_sites(
        self,
        region,
        field,
        site_mask=None,
        inline_array=True,
        chunks="native",
    ):
        """Access SNP site data (positions and alleles).

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Contig name (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        field : {"POS", "REF", "ALT"}
            Array to access.
        site_mask : str, optional
            Site filters mask to apply, e.g. "gamb_colu"
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr
            chunks. Also, can be a target size, e.g., '200 MiB'.

        Returns
        -------
        d : dask.array.Array
            An array of either SNP positions, reference alleles or alternate
            alleles.

        """
        debug = self._log.debug

        region = self.resolve_region(region)
        if isinstance(region, Region):
            region = [region]

        debug("access SNP sites and concatenate over regions")
        ret = da.concatenate(
            [
                self._snp_sites(
                    region=r,
                    field=field,
                    chunks=chunks,
                    inline_array=inline_array,
                )
                for r in region
            ],
            axis=0,
        )

        debug("apply site mask if requested")
        if site_mask is not None:
            loc_sites = self.site_filters(
                region=region,
                mask=site_mask,
                chunks=chunks,
                inline_array=inline_array,
            )
            ret = da_compress(loc_sites, ret, axis=0)

        return ret

    def geneset(self, *args, **kwargs):
        """Deprecated, this method has been renamed to genome_features()."""
        return self.genome_features(*args, **kwargs)

    def resolve_region(self, region):
        """Convert a genome region into a standard data structure.

        Parameters
        ----------
        region: str
            Contig name (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").

        Returns
        -------
        out : Region
            A named tuple with attributes contig, start and end.

        """

        return resolve_region(self, region)

    def open_snp_genotypes(self, sample_set):
        """Open SNP genotypes zarr.

        Parameters
        ----------
        sample_set : str

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        try:
            return self._cache_snp_genotypes[sample_set]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            release_path = self._release_to_path(release)
            path = f"{self._base_path}/{release_path}/snp_genotypes/all/{sample_set}/"
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_snp_genotypes[sample_set] = root
            return root

    def _snp_genotypes(self, *, region, sample_set, field, inline_array, chunks):
        """Access SNP genotypes for a single contig and a single sample set."""
        assert isinstance(region, Region)
        assert isinstance(sample_set, str)
        root = self.open_snp_genotypes(sample_set=sample_set)
        z = root[f"{region.contig}/calldata/{field}"]
        d = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        if region.start or region.end:
            pos = self.snp_sites(region=region.contig, field="POS")
            loc_region = locate_region(region, pos)
            d = d[loc_region]

        return d

    def snp_genotypes(
        self,
        region,
        sample_sets=None,
        sample_query=None,
        field="GT",
        site_mask=None,
        inline_array=True,
        chunks="native",
    ):
        """Access SNP genotypes and associated data.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Contig name (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        field : {"GT", "GQ", "AD", "MQ"}
            Array to access.
        site_mask : str, optional
            Site filters mask to apply, e.g. "gamb_colu"
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr
            chunks. Also, can be a target size, e.g., '200 MiB'.

        Returns
        -------
        d : dask.array.Array
            An array of either genotypes (GT), genotype quality (GQ), allele
            depths (AD) or mapping quality (MQ) values.

        """
        debug = self._log.debug

        debug("normalise parameters")
        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)
        region = self.resolve_region(region)

        debug("normalise region to list to simplify concatenation logic")
        if isinstance(region, Region):
            region = [region]

        debug("concatenate multiple sample sets and/or contigs")
        lx = []
        for r in region:
            ly = []

            for s in sample_sets:
                y = self._snp_genotypes(
                    region=Region(r.contig, None, None),
                    sample_set=s,
                    field=field,
                    inline_array=inline_array,
                    chunks=chunks,
                )
                ly.append(y)

            debug("concatenate data from multiple sample sets")
            x = da.concatenate(ly, axis=1)

            debug("locate region - do this only once, optimisation")
            if r.start or r.end:
                pos = self.snp_sites(region=r.contig, field="POS")
                loc_region = locate_region(r, pos)
                x = x[loc_region]

            lx.append(x)

        debug("concatenate data from multiple regions")
        d = da.concatenate(lx, axis=0)

        debug("apply site filters if requested")
        if site_mask is not None:
            loc_sites = self.site_filters(
                region=region,
                mask=site_mask,
            )
            d = da_compress(loc_sites, d, axis=0)

        debug("apply sample query if requested")
        if sample_query is not None:
            df_samples = self.sample_metadata(sample_sets=sample_sets)
            loc_samples = df_samples.eval(sample_query).values
            d = da.compress(loc_samples, d, axis=1)

        return d

    def open_genome(self):
        """Open the reference genome zarr.

        Returns
        -------
        root : zarr.hierarchy.Group
            Zarr hierarchy containing the reference genome sequence.

        """
        if self._cache_genome is None:
            path = f"{self._base_path}/{self._genome_zarr_path}"
            store = init_zarr_store(fs=self._fs, path=path)
            self._cache_genome = zarr.open_consolidated(store=store)
        return self._cache_genome

    def genome_sequence(self, region, inline_array=True, chunks="native"):
        """Access the reference genome sequence.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Contig name (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr
            chunks. Also, can be a target size, e.g., '200 MiB'.

        Returns
        -------
        d : dask.array.Array
            An array of nucleotides giving the reference genome sequence for the
            given contig.

        """
        genome = self.open_genome()
        region = self.resolve_region(region)
        z = genome[region.contig]
        d = da_from_zarr(z, inline_array=inline_array, chunks=chunks)

        if region.start:
            slice_start = region.start - 1
        else:
            slice_start = None
        if region.end:
            slice_stop = region.end
        else:
            slice_stop = None
        loc_region = slice(slice_start, slice_stop)

        return d[loc_region]

    def is_accessible(
        self,
        region,
        site_mask,
    ):
        """Compute genome accessibility array.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Contig name (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        site_mask : str
            Site filters mask to apply, e.g. "gamb_colu"

        Returns
        -------
        a : numpy.ndarray
            An array of boolean values identifying accessible genome sites.

        """
        debug = self._log.debug

        debug("resolve region")
        region = self.resolve_region(region)

        debug("determine contig sequence length")
        seq_length = self.genome_sequence(region).shape[0]

        debug("set up output")
        is_accessible = np.zeros(seq_length, dtype=bool)

        pos = self.snp_sites(region=region, field="POS").compute()
        if region.start:
            offset = region.start
        else:
            offset = 1

        debug("access site filters")
        filter_pass = self.site_filters(
            region=region,
            mask=site_mask,
        ).compute()

        debug("assign values from site filters")
        is_accessible[pos - offset] = filter_pass

        return is_accessible

    def _snp_df(self, *, transcript):
        """Set up a dataframe with SNP site and filter columns."""
        debug = self._log.debug

        debug("get feature direct from genome_features")
        gs = self.genome_features()
        feature = gs[gs["ID"] == transcript].squeeze()
        contig = feature.contig
        region = Region(contig, feature.start, feature.end)

        debug("grab pos, ref and alt for chrom arm from snp_sites")
        pos = self.snp_sites(region=contig, field="POS")
        ref = self.snp_sites(region=contig, field="REF")
        alt = self.snp_sites(region=contig, field="ALT")
        loc_feature = locate_region(region, pos)
        pos = pos[loc_feature].compute()
        ref = ref[loc_feature].compute()
        alt = alt[loc_feature].compute()

        debug("access site filters")
        filter_pass = dict()
        masks = self._site_mask_ids()
        for m in masks:
            x = self.site_filters(region=contig, mask=m)
            x = x[loc_feature].compute()
            filter_pass[m] = x

        debug("set up columns with contig, pos, ref, alt columns")
        cols = {
            "contig": contig,
            "position": np.repeat(pos, 3),
            "ref_allele": np.repeat(ref.astype("U1"), 3),
            "alt_allele": alt.astype("U1").flatten(),
        }

        debug("add mask columns")
        for m in masks:
            x = filter_pass[m]
            cols[f"pass_{m}"] = np.repeat(x, 3)

        debug("construct dataframe")
        df_snps = pd.DataFrame(cols)

        return region, df_snps

    def _annotator(self):
        """Set up variant effect annotator."""
        if self._cache_annotator is None:
            self._cache_annotator = veff.Annotator(
                genome=self.open_genome(), genome_features=self.genome_features()
            )
        return self._cache_annotator

    def snp_effects(
        self,
        transcript,
        site_mask=None,
    ):
        """Compute variant effects for a gene transcript.

        Parameters
        ----------
        transcript : str
            Gene transcript ID (AgamP4.12), e.g., "AGAP004707-RA".
        site_mask : str, optional
            Site filters mask to apply, e.g. "gamb_colu"

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of all possible SNP variants and their effects, one row
            per variant.

        """
        debug = self._log.debug

        debug("setup initial dataframe of SNPs")
        _, df_snps = self._snp_df(transcript=transcript)

        debug("setup variant effect annotator")
        ann = self._annotator()

        debug("apply mask if requested")
        if site_mask is not None:
            loc_sites = df_snps[f"pass_{site_mask}"]
            df_snps = df_snps.loc[loc_sites]

        debug("reset index after filtering")
        df_snps.reset_index(inplace=True, drop=True)

        debug("add effects to the dataframe")
        ann.get_effects(transcript=transcript, variants=df_snps)

        return df_snps

    def _snp_variants_dataset(self, *, contig, inline_array, chunks):
        debug = self._log.debug

        coords = dict()
        data_vars = dict()

        debug("variant arrays")
        sites_root = self.open_snp_sites()

        debug("variant_position")
        pos_z = sites_root[f"{contig}/variants/POS"]
        variant_position = da_from_zarr(pos_z, inline_array=inline_array, chunks=chunks)
        coords["variant_position"] = [DIM_VARIANT], variant_position

        debug("variant_allele")
        ref_z = sites_root[f"{contig}/variants/REF"]
        alt_z = sites_root[f"{contig}/variants/ALT"]
        ref = da_from_zarr(ref_z, inline_array=inline_array, chunks=chunks)
        alt = da_from_zarr(alt_z, inline_array=inline_array, chunks=chunks)
        variant_allele = da.concatenate([ref[:, None], alt], axis=1)
        data_vars["variant_allele"] = [DIM_VARIANT, DIM_ALLELE], variant_allele

        debug("variant_contig")
        contig_index = self.contigs.index(contig)
        variant_contig = da.full_like(
            variant_position, fill_value=contig_index, dtype="u1"
        )
        coords["variant_contig"] = [DIM_VARIANT], variant_contig

        debug("site filters arrays")
        for mask in self._site_mask_ids():
            filters_root = self.open_site_filters(mask=mask)
            z = filters_root[f"{contig}/variants/filter_pass"]
            d = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
            data_vars[f"variant_filter_pass_{mask}"] = [DIM_VARIANT], d

        debug("set up attributes")
        attrs = {"contigs": self.contigs}

        debug("create a dataset")
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        return ds

    def _snp_calls_dataset(self, *, contig, sample_set, inline_array, chunks):
        debug = self._log.debug

        coords = dict()
        data_vars = dict()

        debug("call arrays")
        calls_root = self.open_snp_genotypes(sample_set=sample_set)
        gt_z = calls_root[f"{contig}/calldata/GT"]
        call_genotype = da_from_zarr(gt_z, inline_array=inline_array, chunks=chunks)
        gq_z = calls_root[f"{contig}/calldata/GQ"]
        call_gq = da_from_zarr(gq_z, inline_array=inline_array, chunks=chunks)
        ad_z = calls_root[f"{contig}/calldata/AD"]
        call_ad = da_from_zarr(ad_z, inline_array=inline_array, chunks=chunks)
        mq_z = calls_root[f"{contig}/calldata/MQ"]
        call_mq = da_from_zarr(mq_z, inline_array=inline_array, chunks=chunks)
        data_vars["call_genotype"] = (
            [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY],
            call_genotype,
        )
        data_vars["call_GQ"] = ([DIM_VARIANT, DIM_SAMPLE], call_gq)
        data_vars["call_MQ"] = ([DIM_VARIANT, DIM_SAMPLE], call_mq)
        data_vars["call_AD"] = (
            [DIM_VARIANT, DIM_SAMPLE, DIM_ALLELE],
            call_ad,
        )

        debug("sample arrays")
        z = calls_root["samples"]
        sample_id = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        # decode to str, as it is stored as bytes objects
        sample_id = sample_id.astype("U")
        coords["sample_id"] = [DIM_SAMPLE], sample_id

        debug("create a dataset")
        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        return ds

    def snp_calls(
        self,
        region,
        sample_sets=None,
        sample_query=None,
        site_mask=None,
        site_class=None,
        inline_array=True,
        chunks="native",
        cohort_size=None,
        random_seed=42,
    ):
        """Access SNP sites, site filters and genotype calls.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Contig name (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "country == 'Burkina Faso'".
        site_mask : str, optional
            Site filters mask to apply, e.g. "gamb_colu"
        site_class : str, optional
            Select sites belonging to one of the following classes: CDS_DEG_4,
            (4-fold degenerate coding sites), CDS_DEG_2_SIMPLE (2-fold simple
            degenerate coding sites), CDS_DEG_0 (non-degenerate coding sites),
            INTRON_SHORT (introns shorter than 100 bp), INTRON_LONG (introns
            longer than 200 bp), INTRON_SPLICE_5PRIME (intron within 2 bp of
            5' splice site), INTRON_SPLICE_3PRIME (intron within 2 bp of 3'
            splice site), UTR_5PRIME (5' untranslated region), UTR_3PRIME (3'
            untranslated region), INTERGENIC (intergenic, more than 10 kbp from
            a gene).
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr
            chunks. Also, can be a target size, e.g., '200 MiB'.
        cohort_size : int, optional
            If provided, randomly down-sample to the given cohort size.
        random_seed : int, optional
            Random seed used for down-sampling.

        Returns
        -------
        ds : xarray.Dataset
            A dataset containing SNP sites, site filters and genotype calls.

        """
        debug = self._log.debug

        debug("normalise parameters")
        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)
        region = self.resolve_region(region)
        if isinstance(region, Region):
            region = [region]

        debug("access SNP calls and concatenate multiple sample sets and/or regions")
        lx = []
        for r in region:

            ly = []
            for s in sample_sets:
                y = self._snp_calls_dataset(
                    contig=r.contig,
                    sample_set=s,
                    inline_array=inline_array,
                    chunks=chunks,
                )
                ly.append(y)

            debug("concatenate data from multiple sample sets")
            x = xarray_concat(ly, dim=DIM_SAMPLE)

            debug("add variants variables")
            v = self._snp_variants_dataset(
                contig=r.contig, inline_array=inline_array, chunks=chunks
            )
            x = xr.merge([v, x], compat="override", join="override")

            debug("handle site class")
            if site_class is not None:
                loc_ann = self._locate_site_class(
                    region=r.contig,
                    site_class=site_class,
                    site_mask=None,
                )
                x = x.isel(variants=loc_ann)

            debug("handle region, do this only once - optimisation")
            if r.start or r.end:
                pos = x["variant_position"].values
                loc_region = locate_region(r, pos)
                x = x.isel(variants=loc_region)

            lx.append(x)

        debug("concatenate data from multiple regions")
        ds = xarray_concat(lx, dim=DIM_VARIANT)

        debug("apply site filters")
        if site_mask is not None:
            ds = dask_compress_dataset(
                ds, indexer=f"variant_filter_pass_{site_mask}", dim=DIM_VARIANT
            )

        debug("add call_genotype_mask")
        ds["call_genotype_mask"] = ds["call_genotype"] < 0

        debug("handle sample query")
        if sample_query is not None:
            if isinstance(sample_query, str):
                df_samples = self.sample_metadata(sample_sets=sample_sets)
                loc_samples = df_samples.eval(sample_query).values
                if np.count_nonzero(loc_samples) == 0:
                    raise ValueError(f"No samples found for query {sample_query!r}")
            else:
                # assume sample query is an indexer, e.g., a list of integers
                loc_samples = sample_query
            ds = ds.isel(samples=loc_samples)

        debug("handle cohort size")
        if cohort_size is not None:
            n_samples = ds.dims["samples"]
            if n_samples < cohort_size:
                raise ValueError(
                    f"not enough samples ({n_samples}) for cohort size ({cohort_size})"
                )
            rng = np.random.default_rng(seed=random_seed)
            loc_downsample = rng.choice(n_samples, size=cohort_size, replace=False)
            loc_downsample.sort()
            ds = ds.isel(samples=loc_downsample)

        return ds

    def snp_dataset(self, *args, **kwargs):
        """Deprecated, this method has been renamed to snp_calls()."""
        return self.snp_calls(*args, **kwargs)

    def _plot_genes_setup_data(self, *, region):
        # default implementation - can be overridden if different attributes
        # used in GFF

        df_genome_features = self.genome_features(
            region=region, attributes=["ID", "Name", "Parent", "description"]
        )
        data = df_genome_features.query("type == 'gene'").copy()

        tooltips = [
            ("ID", "@ID"),
            ("Name", "@Name"),
            ("Description", "@description"),
            ("Location", "@contig:@start{,}-@end{,}"),
        ]

        return data, tooltips

    def plot_genes(
        self,
        region,
        sizing_mode=DEFAULT_GENOME_PLOT_SIZING_MODE,
        width=DEFAULT_GENOME_PLOT_WIDTH,
        height=DEFAULT_GENES_TRACK_HEIGHT,
        show=True,
        toolbar_location="above",
        x_range=None,
        title="Genes",
    ):
        """Plot a genes track, using bokeh.

        Parameters
        ----------
        region : str or Region
            Contig name (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        sizing_mode : str, optional
            Bokeh plot sizing mode, see https://docs.bokeh.org/en/latest/docs/user_guide/layout.html#sizing-modes
        width : int, optional
            Plot width in pixels (px).
        height : int, optional
            Plot height in pixels (px).
        show : bool, optional
            If true, show the plot.
        toolbar_location : str, optional
            Location of bokeh toolbar.
        x_range : bokeh.models.Range1d, optional
            X axis range (for linking to other tracks).
        title : str, optional
            Plot title.

        Returns
        -------
        fig : Figure
            Bokeh figure.

        """
        debug = self._log.debug

        import bokeh.models as bkmod
        import bokeh.plotting as bkplt

        debug("handle region parameter - this determines the genome region to plot")
        region = self.resolve_region(region)
        contig = region.contig
        start = region.start
        end = region.end
        if start is None:
            start = 0
        if end is None:
            end = len(self.genome_sequence(contig))

        debug("define x axis range")
        if x_range is None:
            x_range = bkmod.Range1d(start, end, bounds="auto")

        debug("select the genes overlapping the requested region")
        data, tooltips = self._plot_genes_setup_data(region=region)

        debug(
            "we're going to plot each gene as a rectangle, so add some additional columns"
        )
        data["bottom"] = np.where(data["strand"] == "+", 1, 0)
        data["top"] = data["bottom"] + 0.8

        debug("tidy up missing values for presentation")
        data.fillna("", inplace=True)

        debug("make a figure")
        xwheel_zoom = bkmod.WheelZoomTool(dimensions="width", maintain_focus=False)
        fig = bkplt.figure(
            title=title,
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            tools=[
                "xpan",
                "xzoom_in",
                "xzoom_out",
                xwheel_zoom,
                "reset",
                "tap",
                "hover",
            ],
            toolbar_location=toolbar_location,
            active_scroll=xwheel_zoom,
            active_drag="xpan",
            tooltips=tooltips,
            x_range=x_range,
        )

        debug("add functionality to click through to vectorbase")
        url = "https://vectorbase.org/vectorbase/app/record/gene/@ID"
        taptool = fig.select(type=bkmod.TapTool)
        taptool.callback = bkmod.OpenURL(url=url)

        debug("now plot the genes as rectangles")
        fig.quad(
            bottom="bottom",
            top="top",
            left="start",
            right="end",
            source=data,
            line_width=0.5,
            fill_alpha=0.5,
        )

        debug("tidy up the plot")
        fig.y_range = bkmod.Range1d(-0.4, 2.2)
        fig.ygrid.visible = False
        yticks = [0.4, 1.4]
        yticklabels = ["-", "+"]
        fig.yaxis.ticker = yticks
        fig.yaxis.major_label_overrides = {k: v for k, v in zip(yticks, yticklabels)}
        self._bokeh_style_genome_xaxis(fig, region.contig)

        if show:
            bkplt.show(fig)

        return fig

    def _snp_allele_counts(
        self,
        *,
        region,
        sample_sets,
        sample_query,
        site_mask,
        site_class,
        cohort_size,
        random_seed,
    ):
        debug = self._log.debug

        debug("access SNP calls")
        ds_snps = self.snp_calls(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            site_mask=site_mask,
            site_class=site_class,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )
        gt = ds_snps["call_genotype"]

        debug("set up and run allele counts computation")
        gt = allel.GenotypeDaskArray(gt.data)
        ac = gt.count_alleles(max_allele=3)
        with self._dask_progress(desc="Compute SNP allele counts"):
            ac = ac.compute()

        debug("return plain numpy array")
        results = dict(ac=ac.values)

        return results

    def _dask_progress(self, **kwargs):
        disable = not self._show_progress
        return TqdmCallback(disable=disable, **kwargs)

    def snp_variants(
        self,
        region,
        site_mask=None,
        inline_array=True,
        chunks="native",
    ):
        """Access SNP sites and site filters.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Contig name (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        site_mask : str, optional
            Site filters mask to apply, e.g. "gamb_colu"
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr
            chunks. Also, can be a target size, e.g., '200 MiB'.

        Returns
        -------
        ds : xarray.Dataset
            A dataset containing SNP sites and site filters.

        """
        debug = self._log.debug

        debug("normalise parameters")
        region = self.resolve_region(region)
        if isinstance(region, Region):
            region = [region]

        debug("access SNP data and concatenate multiple regions")
        lx = []
        for r in region:

            debug("access variants")
            x = self._snp_variants_dataset(
                contig=r.contig,
                inline_array=inline_array,
                chunks=chunks,
            )

            debug("handle region")
            if r.start or r.end:
                pos = x["variant_position"].values
                loc_region = locate_region(r, pos)
                x = x.isel(variants=loc_region)

            lx.append(x)

        debug("concatenate data from multiple regions")
        ds = xarray_concat(lx, dim=DIM_VARIANT)

        debug("apply site filters")
        if site_mask is not None:
            ds = dask_compress_dataset(
                ds, indexer=f"variant_filter_pass_{site_mask}", dim=DIM_VARIANT
            )

        return ds

    def plot_transcript(
        self,
        transcript,
        sizing_mode=DEFAULT_GENOME_PLOT_SIZING_MODE,
        width=DEFAULT_GENOME_PLOT_WIDTH,
        height=DEFAULT_GENES_TRACK_HEIGHT,
        show=True,
        x_range=None,
        toolbar_location="above",
        title=True,
    ):
        """Plot a transcript, using bokeh.

        Parameters
        ----------
        transcript : str
            Transcript identifier, e.g., "AGAP004707-RD".
        sizing_mode : str, optional
            Bokeh plot sizing mode, see https://docs.bokeh.org/en/latest/docs/user_guide/layout.html#sizing-modes
        width : int, optional
            Plot width in pixels (px).
        height : int, optional
            Plot height in pixels (px).
        show : bool, optional
            If true, show the plot.
        toolbar_location : str, optional
            Location of bokeh toolbar.
        x_range : bokeh.models.Range1d, optional
            X axis range (for linking to other tracks).
        title : str, optional
            Plot title.

        Returns
        -------
        fig : Figure
            Bokeh figure.

        """
        debug = self._log.debug

        import bokeh.models as bkmod
        import bokeh.plotting as bkplt

        debug("find the transcript annotation")
        df_genome_features = self.genome_features().set_index("ID")
        parent = df_genome_features.loc[transcript]

        if title is True:
            title = f"{transcript} ({parent.strand})"

        debug("define tooltips for hover")
        tooltips = [
            ("Type", "@type"),
            ("Location", "@contig:@start{,}-@end{,}"),
        ]

        debug("make a figure")
        xwheel_zoom = bkmod.WheelZoomTool(dimensions="width", maintain_focus=False)
        fig = bkplt.figure(
            title=title,
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            tools=["xpan", "xzoom_in", "xzoom_out", xwheel_zoom, "reset", "hover"],
            toolbar_location=toolbar_location,
            active_scroll=xwheel_zoom,
            active_drag="xpan",
            tooltips=tooltips,
            x_range=x_range,
        )

        debug("find child components of the transcript")
        data = df_genome_features.set_index("Parent").loc[transcript].copy()
        data["bottom"] = -0.4
        data["top"] = 0.4

        debug("plot exons")
        exons = data.query("type == 'exon'")
        fig.quad(
            bottom="bottom",
            top="top",
            left="start",
            right="end",
            source=exons,
            fill_color=None,
            line_color="black",
            line_width=0.5,
            fill_alpha=0,
        )

        debug("plot introns")
        for intron_start, intron_end in zip(exons[:-1]["end"], exons[1:]["start"]):
            intron_midpoint = (intron_start + intron_end) / 2
            line_data = pd.DataFrame(
                {
                    "x": [intron_start, intron_midpoint, intron_end],
                    "y": [0, 0.1, 0],
                    "type": "intron",
                    "contig": parent.contig,
                    "start": intron_start,
                    "end": intron_end,
                }
            )
            fig.line(
                x="x",
                y="y",
                source=line_data,
                line_width=1,
                line_color="black",
            )

        debug("plot UTRs")
        fig.quad(
            bottom="bottom",
            top="top",
            left="start",
            right="end",
            source=data.query("type == 'five_prime_UTR'"),
            fill_color="green",
            line_width=0,
            fill_alpha=0.5,
        )
        fig.quad(
            bottom="bottom",
            top="top",
            left="start",
            right="end",
            source=data.query("type == 'three_prime_UTR'"),
            fill_color="red",
            line_width=0,
            fill_alpha=0.5,
        )

        debug("plot CDSs")
        fig.quad(
            bottom="bottom",
            top="top",
            left="start",
            right="end",
            source=data.query("type == 'CDS'"),
            fill_color="blue",
            line_width=0,
            fill_alpha=0.5,
        )

        debug("tidy up the figure")
        fig.yaxis.ticker = []
        fig.y_range = bkmod.Range1d(-0.6, 0.6)
        self._bokeh_style_genome_xaxis(fig, parent.contig)

        if show:
            bkplt.show(fig)

        return fig

    def igv(self, region, tracks=None):
        """Create an IGV browser and display it within the notebook.

        Parameters
        ----------
        region: str or Region
            Genomic region defined with coordinates, e.g., "2L:2422600-2422700".
        tracks : list of dict, optional
            Configuration for any additional tracks.

        Returns
        -------
        browser : igv_notebook.Browser

        """
        debug = self._log.debug

        debug("resolve region")
        region = self.resolve_region(region)

        import igv_notebook

        config = {
            "reference": {
                "id": self._genome_ref_id,
                "name": self._genome_ref_name,
                "fastaURL": f"{self._gcs_url}{self._genome_fasta_path}",
                "indexURL": f"{self._gcs_url}{self._genome_fai_path}",
                "tracks": [
                    {
                        "name": "Genes",
                        "type": "annotation",
                        "format": "gff3",
                        "url": f"{self._gcs_url}{self._geneset_gff3_path}",
                        "indexed": False,
                    }
                ],
            },
            "locus": str(region),
        }
        if tracks:
            config["tracks"] = tracks

        debug(config)

        igv_notebook.init()
        browser = igv_notebook.Browser(config)

        return browser

    def view_alignments(
        self,
        region,
        sample,
        visibility_window=20_000,
    ):
        """Launch IGV and view sequence read alignments and SNP genotypes from
        the given sample.

        Parameters
        ----------
        region: str or Region
            Genomic region defined with coordinates, e.g., "2L:2422600-2422700".
        sample : str
            Sample identifier, e.g., "AR0001-C".
        visibility_window : int, optional
            Zoom level in base pairs at which alignment and SNP data will become
            visible.

        Notes
        -----
        Only samples from the Ag3.0 release are currently supported.

        """
        debug = self._log.debug

        debug("look up sample set for sample")
        sample_rec = self.sample_metadata().set_index("sample_id").loc[sample]
        sample_set = sample_rec["sample_set"]

        debug("load data catalog")
        df_cat = self.wgs_data_catalog(sample_set=sample_set)

        debug("locate record for sample")
        cat_rec = df_cat.set_index("sample_id").loc[sample]
        bam_url = cat_rec["alignments_bam"]
        vcf_url = cat_rec["snp_genotypes_vcf"]
        debug(bam_url)
        debug(vcf_url)

        debug("parse region")
        region = self.resolve_region(region)
        contig = region.contig

        # begin creating tracks
        tracks = []

        # https://github.com/igvteam/igv-notebook/issues/3 -- resolved now
        debug("set up site filters tracks")
        self._view_alignments_add_site_filters_tracks(
            contig=contig, visibility_window=visibility_window, tracks=tracks
        )

        debug("add SNPs track")
        tracks.append(
            {
                "name": "SNPs",
                "url": vcf_url,
                "indexURL": f"{vcf_url}.tbi",
                "format": "vcf",
                "type": "variant",
                "visibilityWindow": visibility_window,  # bp
                "height": 50,
            }
        )

        debug("add alignments track")
        tracks.append(
            {
                "name": "Alignments",
                "url": bam_url,
                "indexURL": f"{bam_url}.bai",
                "format": "bam",
                "type": "alignment",
                "visibilityWindow": visibility_window,  # bp
                "height": 500,
            }
        )

        debug("create IGV browser")
        self.igv(region=region, tracks=tracks)

    def _pca(
        self,
        *,
        region,
        n_snps,
        thin_offset,
        sample_sets,
        sample_query,
        site_mask,
        min_minor_ac,
        max_missing_an,
        n_components,
    ):
        debug = self._log.debug

        debug("access SNP calls")
        ds_snps = self.snp_calls(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            site_mask=site_mask,
        )
        debug(
            f"{ds_snps.dims['variants']:,} variants, {ds_snps.dims['samples']:,} samples"
        )

        debug("perform allele count")
        # FIXME: Parameter 'cohort_size', 'random_seed', 'site_class' unfilled
        ac = self.snp_allele_counts(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            site_mask=site_mask,
        )
        n_chroms = ds_snps.dims["samples"] * 2
        an_called = ac.sum(axis=1)
        an_missing = n_chroms - an_called

        debug("ascertain sites")
        ac = allel.AlleleCountsArray(ac)
        min_ref_ac = min_minor_ac
        max_ref_ac = n_chroms - min_minor_ac
        # here we choose biallelic sites involving the reference allele
        loc_sites = (
            ac.is_biallelic()
            & (ac[:, 0] >= min_ref_ac)
            & (ac[:, 0] <= max_ref_ac)
            & (an_missing <= max_missing_an)
        )
        debug(f"ascertained {np.count_nonzero(loc_sites):,} sites")

        debug("thin sites to approximately desired number")
        loc_sites = np.nonzero(loc_sites)[0]
        thin_step = max(loc_sites.shape[0] // n_snps, 1)
        loc_sites_thinned = loc_sites[thin_offset::thin_step]
        debug(f"thinned to {np.count_nonzero(loc_sites_thinned):,} sites")

        debug("access genotypes")
        gt = ds_snps["call_genotype"].data
        gt_asc = da.take(gt, loc_sites_thinned, axis=0)
        gn_asc = allel.GenotypeDaskArray(gt_asc).to_n_alt()
        with self._dask_progress(desc="Load SNP genotypes"):
            gn_asc = gn_asc.compute()

        debug("remove any sites where all genotypes are identical")
        loc_var = np.any(gn_asc != gn_asc[:, 0, np.newaxis], axis=1)
        gn_var = np.compress(loc_var, gn_asc, axis=0)
        debug(f"final shape {gn_var.shape}")

        debug("run the PCA")
        coords, model = allel.pca(gn_var, n_components=n_components)

        debug("work around sign indeterminacy")
        for i in range(n_components):
            c = coords[:, i]
            if np.abs(c.min()) > np.abs(c.max()):
                coords[:, i] = c * -1

        results = dict(coords=coords, evr=model.explained_variance_ratio_)
        return results

    def plot_pca_variance(self, evr, width=900, height=400, **kwargs):
        """Plot explained variance ratios from a principal components analysis
        (PCA) using a plotly bar plot.

        Parameters
        ----------
        evr : np.ndarray
            An array of explained variance ratios, one per component.
        width : int, optional
            Plot width in pixels (px).
        height : int, optional
            Plot height in pixels (px).
        **kwargs
            Passed through to px.bar().

        Returns
        -------
        fig : Figure
            A plotly figure.

        """
        debug = self._log.debug

        import plotly.express as px

        debug("prepare plotting variables")
        y = evr * 100  # convert to percent
        x = [str(i + 1) for i in range(len(y))]

        debug("set up plotting options")
        plot_kwargs = dict(
            labels={
                "x": "Principal component",
                "y": "Explained variance (%)",
            },
            template="simple_white",
            width=width,
            height=height,
        )
        debug("apply any user overrides")
        plot_kwargs.update(kwargs)

        debug("make a bar plot")
        fig = px.bar(x=x, y=y, **plot_kwargs)

        return fig

    def _cohort_alt_allele_counts_melt(self, gt, indices, max_allele):
        ac_alt_melt, an = self._cohort_alt_allele_counts_melt_kernel(
            gt, indices, max_allele
        )
        an_melt = np.repeat(an, max_allele, axis=0)
        return ac_alt_melt, an_melt

    def _prep_samples_for_cohort_grouping(self, *, df_samples, area_by, period_by):

        # take a copy, as we will modify the dataframe
        df_samples = df_samples.copy()

        # fix intermediate taxon values - we only want to build cohorts with clean
        # taxon calls, so we set intermediate values to None
        loc_intermediate_taxon = (
            df_samples["taxon"].str.startswith("intermediate").fillna(False)
        )
        df_samples.loc[loc_intermediate_taxon, "taxon"] = None

        # add period column
        if period_by == "year":
            make_period = self._make_sample_period_year
        elif period_by == "quarter":
            make_period = self._make_sample_period_quarter
        elif period_by == "month":
            make_period = self._make_sample_period_month
        else:
            raise ValueError(
                f"Value for period_by parameter must be one of 'year', 'quarter', 'month'; found {period_by!r}."
            )
        sample_period = df_samples.apply(make_period, axis="columns")
        df_samples["period"] = sample_period

        # add area column for consistent output
        df_samples["area"] = df_samples[area_by]

        return df_samples

    def _region_str(self, region):
        """Convert a region to a string representation.

        Parameters
        ----------
        region : Region or list of Region
            The region to display.

        Returns
        -------
        out : str

        """
        if isinstance(region, list):
            if len(region) > 1:
                return "; ".join([self._region_str(r) for r in region])
            else:
                region = region[0]

        # sanity check
        assert isinstance(region, Region)

        return str(region)

    def genome_features(
        self, region=None, attributes=("ID", "Parent", "Name", "description")
    ):
        """Access genome feature annotations.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Contig name (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        attributes : list of str, optional
            Attribute keys to unpack into columns. Provide "*" to unpack all
            attributes.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of genome annotations, one row per feature.

        """
        debug = self._log.debug

        if attributes is not None:
            attributes = tuple(attributes)

        try:
            df = self._cache_genome_features[attributes]

        except KeyError:
            path = f"{self._base_path}/{self._geneset_gff3_path}"
            with self._fs.open(path, mode="rb") as f:
                df = read_gff3(f, compression="gzip")
            if attributes is not None:
                df = unpack_gff3_attributes(df, attributes=attributes)
            self._cache_genome_features[attributes] = df

        debug("handle region")
        if region is not None:

            region = self.resolve_region(region)

            debug("normalise to list to simplify concatenation logic")
            if isinstance(region, Region):
                region = [region]

            debug("apply region query")
            parts = []
            for r in region:
                df_part = df.query(f"contig == '{r.contig}'")
                if r.end is not None:
                    df_part = df_part.query(f"start <= {r.end}")
                if r.start is not None:
                    df_part = df_part.query(f"end >= {r.start}")
                parts.append(df_part)
            df = pd.concat(parts, axis=0)

        return df.reset_index(drop=True).copy()

    def _lookup_sample(self, sample, sample_set=None):
        df_samples = self.sample_metadata(sample_sets=sample_set).set_index("sample_id")
        sample_rec = None
        if isinstance(sample, str):
            sample_rec = df_samples.loc[sample]
        elif isinstance(sample, int):
            sample_rec = df_samples.iloc[sample]
        else:
            type_error(name="sample", value=sample, expectation=(str, int))
        return sample_rec

    def _plot_heterozygosity_track(
        self,
        *,
        sample_id,
        sample_set,
        windows,
        counts,
        region,
        window_size,
        y_max,
        sizing_mode,
        width,
        height,
        circle_kwargs,
        show,
        x_range,
    ):
        debug = self._log.debug

        import bokeh.models as bkmod
        import bokeh.plotting as bkplt

        region = self.resolve_region(region)

        # pos axis
        window_pos = windows.mean(axis=1)

        # het axis
        window_het = counts / window_size

        # determine plotting limits
        if x_range is None:
            if region.start is not None:
                x_min = region.start
            else:
                x_min = 0
            if region.end is not None:
                x_max = region.end
            else:
                x_max = len(self.genome_sequence(region.contig))
            x_range = bkmod.Range1d(x_min, x_max, bounds="auto")

        debug("create a figure for plotting")
        xwheel_zoom = bkmod.WheelZoomTool(dimensions="width", maintain_focus=False)
        fig = bkplt.figure(
            title=f"{sample_id} ({sample_set})",
            tools=["xpan", "xzoom_in", "xzoom_out", xwheel_zoom, "reset"],
            active_scroll=xwheel_zoom,
            active_drag="xpan",
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            toolbar_location="above",
            x_range=x_range,
            y_range=(0, y_max),
        )

        debug("plot heterozygosity")
        data = pd.DataFrame(
            {
                "position": window_pos,
                "heterozygosity": window_het,
            }
        )
        if circle_kwargs is None:
            circle_kwargs = dict()
        circle_kwargs.setdefault("size", 3)
        fig.circle(x="position", y="heterozygosity", source=data, **circle_kwargs)

        debug("tidy up the plot")
        fig.yaxis.axis_label = "Heterozygosity (bp⁻¹)"
        self._bokeh_style_genome_xaxis(fig, region.contig)

        if show:
            bkplt.show(fig)

        return fig

    def plot_heterozygosity_track(
        self,
        sample,
        region,
        site_mask="default",
        window_size=20_000,
        sample_set=None,
        y_max=0.03,
        sizing_mode=DEFAULT_GENOME_PLOT_SIZING_MODE,
        width=DEFAULT_GENOME_PLOT_WIDTH,
        height=200,
        circle_kwargs=None,
        show=True,
        x_range=None,
    ):
        """Plot windowed heterozygosity for a single sample over a genome
        region.

        Parameters
        ----------
        sample : str or int
            Sample identifier or index within sample set.
        region : str
            Contig name (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        site_mask : str, optional
            Site filters mask to apply, e.g. "gamb_colu"
        window_size : int, optional
            Number of sites per window.
        sample_set : str, optional
            Sample set identifier. Not needed if sample parameter gives a sample
            identifier.
        y_max : float, optional
            Y axis limit.
        sizing_mode : str, optional
            Bokeh plot sizing mode, see https://docs.bokeh.org/en/latest/docs/user_guide/layout.html#sizing-modes
        width : int, optional
            Plot width in pixels (px).
        height : int, optional
            Plot height in pixels (px).
        circle_kwargs : dict, optional
            Passed through to bokeh circle() function.
        show : bool, optional
            If true, show the plot.
        x_range : bokeh.models.Range1d, optional
            X axis range (for linking to other tracks).

        Returns
        -------
        fig : Figure
            Bokeh figure.

        """
        debug = self._log.debug

        debug("compute windowed heterozygosity")
        sample_id, sample_set, windows, counts = self._sample_count_het(
            sample=sample,
            region=region,
            site_mask=site_mask,
            window_size=window_size,
            sample_set=sample_set,
        )

        debug("plot heterozygosity")
        fig = self._plot_heterozygosity_track(
            sample_id=sample_id,
            sample_set=sample_set,
            windows=windows,
            counts=counts,
            region=region,
            window_size=window_size,
            y_max=y_max,
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            circle_kwargs=circle_kwargs,
            show=show,
            x_range=x_range,
        )

        return fig

    def plot_heterozygosity(
        self,
        sample,
        region,
        site_mask="default",
        window_size=20_000,
        sample_set=None,
        y_max=0.03,
        sizing_mode=DEFAULT_GENOME_PLOT_SIZING_MODE,
        width=DEFAULT_GENOME_PLOT_WIDTH,
        track_height=170,
        genes_height=DEFAULT_GENES_TRACK_HEIGHT,
        circle_kwargs=None,
        show=True,
    ):
        """Plot windowed heterozygosity for a single sample over a genome
        region.

        Parameters
        ----------
        sample : str or int
            Sample identifier or index within sample set.
        region : str
            Contig name (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        site_mask : str, optional
            Site filters mask to apply, e.g. "gamb_colu"
        window_size : int, optional
            Number of sites per window.
        sample_set : str, optional
            Sample set identifier. Not needed if sample parameter gives a sample
            identifier.
        y_max : float, optional
            Y axis limit.
        sizing_mode : str, optional
            Bokeh plot sizing mode, see https://docs.bokeh.org/en/latest/docs/user_guide/layout.html#sizing-modes
        width : int, optional
            Plot width in pixels (px).
        track_height : int, optional
            Heterozygosity track height in pixels (px).
        genes_height : int, optional
            Genes track height in pixels (px).
        circle_kwargs : dict, optional
            Passed through to bokeh circle() function.
        show : bool, optional
            If true, show the plot.

        Returns
        -------
        fig : Figure
            Bokeh figure.

        """

        debug = self._log.debug

        import bokeh.layouts as bklay
        import bokeh.plotting as bkplt

        # normalise to support multiple samples
        if isinstance(sample, (list, tuple)):
            samples = sample
        else:
            samples = [sample]

        debug("plot first sample track")
        fig1 = self.plot_heterozygosity_track(
            sample=samples[0],
            sample_set=sample_set,
            region=region,
            site_mask=site_mask,
            window_size=window_size,
            y_max=y_max,
            sizing_mode=sizing_mode,
            width=width,
            height=track_height,
            circle_kwargs=circle_kwargs,
            show=False,
        )
        fig1.xaxis.visible = False
        figs = [fig1]

        debug("plot remaining sample tracks")
        for sample in samples[1:]:
            fig_het = self.plot_heterozygosity_track(
                sample=sample,
                sample_set=sample_set,
                region=region,
                site_mask=site_mask,
                window_size=window_size,
                y_max=y_max,
                sizing_mode=sizing_mode,
                width=width,
                height=track_height,
                circle_kwargs=circle_kwargs,
                show=False,
                x_range=fig1.x_range,
            )
            fig_het.xaxis.visible = False
            figs.append(fig_het)

        debug("plot genes track")
        fig_genes = self.plot_genes(
            region=region,
            sizing_mode=sizing_mode,
            width=width,
            height=genes_height,
            x_range=fig1.x_range,
            show=False,
        )
        figs.append(fig_genes)

        debug("combine plots into a single figure")
        fig_all = bklay.gridplot(
            figs,
            ncols=1,
            toolbar_location="above",
            merge_tools=True,
            sizing_mode=sizing_mode,
        )

        if show:
            bkplt.show(fig_all)

        return fig_all

    def _sample_count_het(
        self,
        sample,
        region,
        site_mask,
        window_size,
        sample_set=None,
    ):
        debug = self._log.debug

        region = self.resolve_region(region)
        if site_mask == "default":
            site_mask = self._default_site_mask

        debug("access sample metadata, look up sample")
        sample_rec = self._lookup_sample(sample=sample, sample_set=sample_set)
        sample_id = sample_rec.name  # sample_id
        sample_set = sample_rec["sample_set"]

        debug("access SNPs, select data for sample")
        ds_snps = self.snp_calls(
            region=region, sample_sets=sample_set, site_mask=site_mask
        )
        ds_snps_sample = ds_snps.set_index(samples="sample_id").sel(samples=sample_id)

        # snp positions
        pos = ds_snps_sample["variant_position"].values

        # access genotypes
        gt = allel.GenotypeDaskVector(ds_snps_sample["call_genotype"].data)

        # compute het
        with self._dask_progress(desc="Compute heterozygous genotypes"):
            is_het = gt.is_het().compute()

        # compute window coordinates
        windows = allel.moving_statistic(
            values=pos,
            statistic=lambda x: [x[0], x[-1]],
            size=window_size,
        )

        # compute windowed heterozygosity
        counts = allel.moving_statistic(
            values=is_het,
            statistic=np.sum,
            size=window_size,
        )

        return sample_id, sample_set, windows, counts

    def roh_hmm(
        self,
        sample,
        region,
        window_size=20_000,
        site_mask="default",
        sample_set=None,
        phet_roh=0.001,
        phet_nonroh=(0.003, 0.01),
        transition=1e-3,
    ):
        """Infer runs of homozygosity for a single sample over a genome region.

        Parameters
        ----------
        sample : str or int
            Sample identifier or index within sample set.
        region : str
            Contig name (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        window_size : int, optional
            Number of sites per window.
        site_mask : str, optional
            Site filters mask to apply, e.g. "gamb_colu"
        sample_set : str, optional
            Sample set identifier. Not needed if sample parameter gives a sample
            identifier.
        phet_roh: float, optional
            Probability of observing a heterozygote in a ROH.
        phet_nonroh: tuple of floats, optional
            One or more probabilities of observing a heterozygote outside a ROH.
        transition: float, optional
           Probability of moving between states. A larger window size may call
           for a larger transitional probability.

        Returns
        -------
        df_roh : pandas.DataFrame
            A DataFrame where each row provides data about a single run of
            homozygosity.

        """
        debug = self._log.debug

        region = self.resolve_region(region)

        debug("compute windowed heterozygosity")
        sample_id, sample_set, windows, counts = self._sample_count_het(
            sample=sample,
            region=region,
            site_mask=site_mask,
            window_size=window_size,
            sample_set=sample_set,
        )

        debug("compute runs of homozygosity")
        df_roh = self._roh_hmm_predict(
            windows=windows,
            counts=counts,
            phet_roh=phet_roh,
            phet_nonroh=phet_nonroh,
            transition=transition,
            window_size=window_size,
            sample_id=sample_id,
            contig=region.contig,
        )

        return df_roh

    def plot_roh_track(
        self,
        df_roh,
        region,
        sizing_mode=DEFAULT_GENOME_PLOT_SIZING_MODE,
        width=DEFAULT_GENOME_PLOT_WIDTH,
        height=100,
        show=True,
        x_range=None,
        title="Runs of homozygosity",
    ):
        debug = self._log.debug

        import bokeh.models as bkmod
        import bokeh.plotting as bkplt

        debug("handle region parameter - this determines the genome region to plot")
        region = self.resolve_region(region)
        contig = region.contig
        start = region.start
        end = region.end
        if start is None:
            start = 0
        if end is None:
            end = len(self.genome_sequence(contig))

        debug("define x axis range")
        if x_range is None:
            x_range = bkmod.Range1d(start, end, bounds="auto")

        debug(
            "we're going to plot each gene as a rectangle, so add some additional columns"
        )
        data = df_roh.copy()
        data["bottom"] = 0.2
        data["top"] = 0.8

        debug("make a figure")
        xwheel_zoom = bkmod.WheelZoomTool(dimensions="width", maintain_focus=False)
        fig = bkplt.figure(
            title=title,
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            tools=[
                "xpan",
                "xzoom_in",
                "xzoom_out",
                xwheel_zoom,
                "reset",
                "tap",
                "hover",
            ],
            active_scroll=xwheel_zoom,
            active_drag="xpan",
            x_range=x_range,
        )

        debug("now plot the ROH as rectangles")
        fig.quad(
            bottom="bottom",
            top="top",
            left="roh_start",
            right="roh_stop",
            source=data,
            line_width=0.5,
            fill_alpha=0.5,
        )

        debug("tidy up the plot")
        fig.y_range = bkmod.Range1d(0, 1)
        fig.ygrid.visible = False
        fig.yaxis.ticker = []
        self._bokeh_style_genome_xaxis(fig, region.contig)

        if show:
            bkplt.show(fig)

        return fig

    def plot_roh(
        self,
        sample,
        region,
        window_size=20_000,
        site_mask="default",
        sample_set=None,
        phet_roh=0.001,
        phet_nonroh=(0.003, 0.01),
        transition=1e-3,
        y_max=0.03,
        sizing_mode=DEFAULT_GENOME_PLOT_SIZING_MODE,
        width=DEFAULT_GENOME_PLOT_WIDTH,
        heterozygosity_height=170,
        roh_height=50,
        genes_height=DEFAULT_GENES_TRACK_HEIGHT,
        circle_kwargs=None,
        show=True,
    ):
        """Plot windowed heterozygosity and inferred runs of homozygosity for a
        single sample over a genome region.

        Parameters
        ----------
        sample : str or int
            Sample identifier or index within sample set.
        region : str
            Contig name (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        window_size : int, optional
            Number of sites per window.
        site_mask : str, optional
            Site filters mask to apply, e.g. "gamb_colu"
        sample_set : str, optional
            Sample set identifier. Not needed if sample parameter gives a sample
            identifier.
        phet_roh: float, optional
            Probability of observing a heterozygote in a ROH.
        phet_nonroh: tuple of floats, optional
            One or more probabilities of observing a heterozygote outside a ROH.
        transition: float, optional
           Probability of moving between states. A larger window size may call
           for a larger transitional probability.
        y_max : float, optional
            Y axis limit.
        sizing_mode : str, optional
            Bokeh plot sizing mode, see https://docs.bokeh.org/en/latest/docs/user_guide/layout.html#sizing-modes
        width : int, optional
            Plot width in pixels (px).
        heterozygosity_height : int, optional
            Heterozygosity track height in pixels (px).
        roh_height : int, optional
            ROH track height in pixels (px).
        genes_height : int, optional
            Genes track height in pixels (px).
        circle_kwargs : dict, optional
            Passed through to bokeh circle() function.
        show : bool, optional
            If true, show the plot.

        Returns
        -------
        fig : Figure
            Bokeh figure.

        """

        debug = self._log.debug

        import bokeh.layouts as bklay
        import bokeh.plotting as bkplt

        region = self.resolve_region(region)

        debug("compute windowed heterozygosity")
        sample_id, sample_set, windows, counts = self._sample_count_het(
            sample=sample,
            region=region,
            site_mask=site_mask,
            window_size=window_size,
            sample_set=sample_set,
        )

        debug("plot_heterozygosity track")
        fig_het = self._plot_heterozygosity_track(
            sample_id=sample_id,
            sample_set=sample_set,
            windows=windows,
            counts=counts,
            region=region,
            window_size=window_size,
            y_max=y_max,
            sizing_mode=sizing_mode,
            width=width,
            height=heterozygosity_height,
            circle_kwargs=circle_kwargs,
            show=False,
            x_range=None,
        )
        fig_het.xaxis.visible = False
        figs = [fig_het]

        debug("compute runs of homozygosity")
        df_roh = self._roh_hmm_predict(
            windows=windows,
            counts=counts,
            phet_roh=phet_roh,
            phet_nonroh=phet_nonroh,
            transition=transition,
            window_size=window_size,
            sample_id=sample_id,
            contig=region.contig,
        )

        debug("plot roh track")
        fig_roh = self.plot_roh_track(
            df_roh,
            region=region,
            sizing_mode=sizing_mode,
            width=width,
            height=roh_height,
            show=False,
            x_range=fig_het.x_range,
        )
        fig_roh.xaxis.visible = False
        figs.append(fig_roh)

        debug("plot genes track")
        fig_genes = self.plot_genes(
            region=region,
            sizing_mode=sizing_mode,
            width=width,
            height=genes_height,
            x_range=fig_het.x_range,
            show=False,
        )
        figs.append(fig_genes)

        debug("combine plots into a single figure")
        fig_all = bklay.gridplot(
            figs,
            ncols=1,
            toolbar_location="above",
            merge_tools=True,
            sizing_mode=sizing_mode,
        )

        if show:
            bkplt.show(fig_all)

        return fig_all

    def _locate_site_class(
        self,
        *,
        region,
        site_mask,
        site_class,
    ):
        debug = self._log.debug

        # cache these data in memory to avoid repeated computation
        cache_key = (region, site_mask, site_class)

        try:
            loc_ann = self._cache_locate_site_class[cache_key]

        except KeyError:
            debug("access site annotations data")
            ds_ann = self.site_annotations(
                region=region,
                site_mask=site_mask,
            )
            codon_pos = ds_ann["codon_position"].data
            codon_deg = ds_ann["codon_degeneracy"].data
            seq_cls = ds_ann["seq_cls"].data
            seq_flen = ds_ann["seq_flen"].data
            seq_relpos_start = ds_ann["seq_relpos_start"].data
            seq_relpos_stop = ds_ann["seq_relpos_stop"].data
            site_class = site_class.upper()

            debug("define constants used in site annotations data")
            # FIXME: variable in function should be lowercase
            SEQ_CLS_UNKNOWN = 0  # noqa
            SEQ_CLS_UPSTREAM = 1
            SEQ_CLS_DOWNSTREAM = 2
            SEQ_CLS_5UTR = 3
            SEQ_CLS_3UTR = 4
            SEQ_CLS_CDS_FIRST = 5
            SEQ_CLS_CDS_MID = 6
            SEQ_CLS_CDS_LAST = 7
            SEQ_CLS_INTRON_FIRST = 8
            SEQ_CLS_INTRON_MID = 9
            SEQ_CLS_INTRON_LAST = 10
            CODON_DEG_UNKNOWN = 0  # noqa
            CODON_DEG_0 = 1
            CODON_DEG_2_SIMPLE = 2
            CODON_DEG_2_COMPLEX = 3  # noqa
            CODON_DEG_4 = 4

            debug("set up site selection")

            if site_class == "CDS_DEG_4":
                # 4-fold degenerate coding sites
                loc_ann = (
                    (
                        (seq_cls == SEQ_CLS_CDS_FIRST)
                        | (seq_cls == SEQ_CLS_CDS_MID)
                        | (seq_cls == SEQ_CLS_CDS_LAST)
                    )
                    & (codon_pos == 2)
                    & (codon_deg == CODON_DEG_4)
                )

            elif site_class == "CDS_DEG_2_SIMPLE":
                # 2-fold degenerate coding sites
                loc_ann = (
                    (
                        (seq_cls == SEQ_CLS_CDS_FIRST)
                        | (seq_cls == SEQ_CLS_CDS_MID)
                        | (seq_cls == SEQ_CLS_CDS_LAST)
                    )
                    & (codon_pos == 2)
                    & (codon_deg == CODON_DEG_2_SIMPLE)
                )

            elif site_class == "CDS_DEG_0":
                # non-degenerate coding sites
                loc_ann = (
                    (seq_cls == SEQ_CLS_CDS_FIRST)
                    | (seq_cls == SEQ_CLS_CDS_MID)
                    | (seq_cls == SEQ_CLS_CDS_LAST)
                ) & (codon_deg == CODON_DEG_0)

            elif site_class == "INTRON_SHORT":
                # short introns, excluding splice regions
                loc_ann = (
                    (
                        (seq_cls == SEQ_CLS_INTRON_FIRST)
                        | (seq_cls == SEQ_CLS_INTRON_MID)
                        | (seq_cls == SEQ_CLS_INTRON_LAST)
                    )
                    & (seq_flen < 100)
                    & (seq_relpos_start > 10)
                    & (seq_relpos_stop > 10)
                )

            elif site_class == "INTRON_LONG":
                # long introns, excluding splice regions
                loc_ann = (
                    (
                        (seq_cls == SEQ_CLS_INTRON_FIRST)
                        | (seq_cls == SEQ_CLS_INTRON_MID)
                        | (seq_cls == SEQ_CLS_INTRON_LAST)
                    )
                    & (seq_flen > 200)
                    & (seq_relpos_start > 10)
                    & (seq_relpos_stop > 10)
                )

            elif site_class == "INTRON_SPLICE_5PRIME":
                # 5' intron splice regions
                loc_ann = (
                    (seq_cls == SEQ_CLS_INTRON_FIRST)
                    | (seq_cls == SEQ_CLS_INTRON_MID)
                    | (seq_cls == SEQ_CLS_INTRON_LAST)
                ) & (seq_relpos_start < 2)

            elif site_class == "INTRON_SPLICE_3PRIME":
                # 3' intron splice regions
                loc_ann = (
                    (seq_cls == SEQ_CLS_INTRON_FIRST)
                    | (seq_cls == SEQ_CLS_INTRON_MID)
                    | (seq_cls == SEQ_CLS_INTRON_LAST)
                ) & (seq_relpos_stop < 2)

            elif site_class == "UTR_5PRIME":
                # 5' UTR
                loc_ann = seq_cls == SEQ_CLS_5UTR

            elif site_class == "UTR_3PRIME":
                # 3' UTR
                loc_ann = seq_cls == SEQ_CLS_3UTR

            elif site_class == "INTERGENIC":
                # intergenic regions, distant from a gene
                loc_ann = (
                    (seq_cls == SEQ_CLS_UPSTREAM) & (seq_relpos_stop > 10_000)
                ) | ((seq_cls == SEQ_CLS_DOWNSTREAM) & (seq_relpos_start > 10_000))

            else:
                raise NotImplementedError(site_class)

            debug("compute site selection")
            with self._dask_progress(desc=f"Locate {site_class} sites"):
                loc_ann = loc_ann.compute()

            self._cache_locate_site_class[cache_key] = loc_ann

        return loc_ann

    def site_annotations(
        self,
        region,
        site_mask=None,
        inline_array=True,
        chunks="auto",
    ):
        """Load site annotations.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Contig name (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        site_mask : str
            Site filters mask to apply, e.g. "gamb_colu"
        inline_array : bool, optional
            Passed through to dask.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr
            chunks. Also, can be a target size, e.g., '200 MiB'.

        Returns
        -------
        ds : xarray.Dataset
            A dataset of site annotations.

        """
        # N.B., we default to chunks="auto" here for performance reasons

        debug = self._log.debug

        debug("resolve region")
        region = self.resolve_region(region)
        if isinstance(region, list):
            raise TypeError("Multiple regions not supported.")
        contig = region.contig

        debug("open site annotations zarr")
        root = self.open_site_annotations()

        debug("build a dataset")
        ds = xr.Dataset()
        for field in (
            "codon_degeneracy",
            "codon_nonsyn",
            "codon_position",
            "seq_cls",
            "seq_flen",
            "seq_relpos_start",
            "seq_relpos_stop",
        ):
            data = da_from_zarr(
                root[field][contig],
                inline_array=inline_array,
                chunks=chunks,
            )
            ds[field] = "variants", data

        debug("subset to SNP positions")
        pos = self.snp_sites(
            region=contig,
            field="POS",
            site_mask=site_mask,
            inline_array=inline_array,
            chunks=chunks,
        )
        pos = pos.compute()
        if region.start or region.end:
            loc_region = locate_region(region, pos)
            pos = pos[loc_region]
        idx = pos - 1
        ds = ds.isel(variants=idx)

        return ds

    def wgs_data_catalog(self, sample_set):
        """Load a data catalog providing URLs for downloading BAM, VCF and Zarr
        files for samples in a given sample set.

        Parameters
        ----------
        sample_set : str
            Sample set identifier.

        Returns
        -------
        df : pandas.DataFrame
            One row per sample, columns provide URLs.

        """
        debug = self._log.debug

        debug("look up release for sample set")
        release = self._lookup_release(sample_set=sample_set)
        release_path = self._release_to_path(release=release)

        debug("load data catalog")
        path = f"{self._base_path}/{release_path}/metadata/general/{sample_set}/wgs_snp_data.csv"
        with self._fs.open(path) as f:
            df = pd.read_csv(f, na_values="")

        debug("normalise columns")
        df = df[
            [
                "sample_id",
                "alignments_bam",
                "snp_genotypes_vcf",
                "snp_genotypes_zarr",
            ]
        ]

        return df

    def pca(
        self,
        region,
        n_snps,
        thin_offset=0,
        sample_sets=None,
        sample_query=None,
        site_mask="default",
        min_minor_ac=2,
        max_missing_an=0,
        n_components=20,
    ):
        """Run a principal components analysis (PCA) using biallelic SNPs from
        the selected genome region and samples.

        Parameters
        ----------
        region : str
            Contig name (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        n_snps : int
            The desired number of SNPs to use when running the analysis.
            SNPs will be evenly thinned to approximately this number.
        thin_offset : int, optional
            Starting index for SNP thinning. Change this to repeat the analysis
            using a different set of SNPs.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "country == 'Burkina Faso'".
        site_mask : str, optional
            Site filters mask to apply, e.g. "gamb_colu"
        min_minor_ac : int, optional
            The minimum minor allele count. SNPs with a minor allele count
            below this value will be excluded prior to thinning.
        max_missing_an : int, optional
            The maximum number of missing allele calls to accept. SNPs with
            more than this value will be excluded prior to thinning. Set to 0
            (default) to require no missing calls.
        n_components : int, optional
            Number of components to return.

        Returns
        -------
        df_pca : pandas.DataFrame
            A dataframe of sample metadata, with columns "PC1", "PC2", "PC3",
            etc., added.
        evr : np.ndarray
            An array of explained variance ratios, one per component.

        Notes
        -----
        This computation may take some time to run, depending on your computing
        environment. Results of this computation will be cached and re-used if
        the `results_cache` parameter was set when instantiating the Ag3 class.

        """
        debug = self._log.debug

        # change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data
        name = self._pca_results_cache_name

        if site_mask == "default":
            site_mask = self._default_site_mask

        debug("normalize params for consistent hash value")
        params = dict(
            region=self.resolve_region(region).to_dict(),
            n_snps=n_snps,
            thin_offset=thin_offset,
            sample_sets=self._prep_sample_sets_arg(sample_sets=sample_sets),
            sample_query=sample_query,
            site_mask=site_mask,
            min_minor_ac=min_minor_ac,
            max_missing_an=max_missing_an,
            n_components=n_components,
        )

        debug("try to retrieve results from the cache")
        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._pca(**params)
            self.results_cache_set(name=name, params=params, results=results)

        debug("unpack results")
        coords = results["coords"]
        evr = results["evr"]

        debug("add coords to sample metadata dataframe")
        df_samples = self.sample_metadata(
            sample_sets=sample_sets,
            sample_query=sample_query,
        )
        df_coords = pd.DataFrame(
            {f"PC{i + 1}": coords[:, i] for i in range(n_components)}
        )
        df_pca = pd.concat([df_samples, df_coords], axis="columns")

        return df_pca, evr

    def plot_snps(
        self,
        region,
        sample_sets=None,
        sample_query=None,
        site_mask="default",
        cohort_size=None,
        sizing_mode=DEFAULT_GENOME_PLOT_SIZING_MODE,
        width=DEFAULT_GENOME_PLOT_WIDTH,
        track_height=80,
        genes_height=DEFAULT_GENES_TRACK_HEIGHT,
        max_snps=200_000,
        show=True,
    ):
        """Plot SNPs in a given genome region. SNPs are shown as rectangles,
        with segregating and non-segregating SNPs positioned on different levels,
        and coloured by site filter.

        Parameters
        ----------
        region : str
            Contig name (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "country == 'Burkina Faso'".
        site_mask : str, optional
            Site filters mask to apply, e.g. "gamb_colu".
        cohort_size : int, optional
            If provided, randomly down-sample to the given cohort size before
            computing allele counts.
        sizing_mode : str, optional
            Bokeh plot sizing mode, see https://docs.bokeh.org/en/latest/docs/user_guide/layout.html#sizing-modes
        width : int, optional
            Width of plot in pixels (px).
        track_height : int, optional
            Height of SNPs track in pixels (px).
        genes_height : int, optional
            Height of genes track in pixels (px).
        max_snps : int, optional
            Maximum number of SNPs to show.
        show : bool, optional
            If True, show the plot.

        Returns
        -------
        fig : Figure
            Bokeh figure.

        """
        debug = self._log.debug

        if site_mask == "default":
            site_mask = self._default_site_mask

        import bokeh.layouts as bklay
        import bokeh.plotting as bkplt

        debug("plot SNPs track")
        fig1 = self.plot_snps_track(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            site_mask=site_mask,
            cohort_size=cohort_size,
            sizing_mode=sizing_mode,
            width=width,
            height=track_height,
            max_snps=max_snps,
            show=False,
        )
        fig1.xaxis.visible = False

        debug("plot genes track")
        fig2 = self.plot_genes(
            region=region,
            sizing_mode=sizing_mode,
            width=width,
            height=genes_height,
            x_range=fig1.x_range,
            show=False,
        )

        fig = bklay.gridplot(
            [fig1, fig2],
            ncols=1,
            toolbar_location="above",
            merge_tools=True,
            sizing_mode=sizing_mode,
        )

        if show:
            bkplt.show(fig)

        return fig

    def open_site_annotations(self):
        """Open site annotations zarr.

        Returns
        -------
        root : zarr.hierarchy.Group

        """

        if self._cache_site_annotations is None:
            path = f"{self._base_path}/{self._site_annotations_zarr_path}"
            store = init_zarr_store(fs=self._fs, path=path)
            self._cache_site_annotations = zarr.open_consolidated(store=store)
        return self._cache_site_annotations

    def plot_snps_track(
        self,
        region,
        sample_sets=None,
        sample_query=None,
        site_mask="default",
        cohort_size=None,
        sizing_mode=DEFAULT_GENOME_PLOT_SIZING_MODE,
        width=DEFAULT_GENOME_PLOT_WIDTH,
        height=120,
        max_snps=200_000,
        x_range=None,
        show=True,
    ):
        """Plot SNPs in a given genome region. SNPs are shown as rectangles,
        with segregating and non-segregating SNPs positioned on different levels,
        and coloured by site filter.

        Parameters
        ----------
        region : str
            Contig name (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "country == 'Burkina Faso'".
        site_mask : str, optional
            Site filters mask to apply, e.g. "gamb_colu".
        cohort_size : int, optional
            If provided, randomly down-sample to the given cohort size before
            computing allele counts.
        sizing_mode : str, optional
            Bokeh plot sizing mode, see https://docs.bokeh.org/en/latest/docs/user_guide/layout.html#sizing-modes
        width : int, optional
            Width of plot in pixels (px).
        height : int, optional
            Height of plot in pixels (px).
        max_snps : int, optional
            Maximum number of SNPs to plot.
        x_range : bokeh.models.Range1d, optional
            X axis range (for linking to other tracks).
        show : bool, optional
            If True, show the plot.

        Returns
        -------
        fig : Figure
            Bokeh figure.

        """
        debug = self._log.debug

        if site_mask == "default":
            site_mask = self._default_site_mask

        import bokeh.models as bkmod
        import bokeh.palettes as bkpal
        import bokeh.plotting as bkplt

        debug("resolve and check region")
        region = self.resolve_region(region)
        if (
            (region.start is None)
            or (region.end is None)
            or ((region.end - region.start) > max_snps)
        ):
            raise ValueError("Region is too large, please provide a smaller region.")

        debug("compute allele counts")
        # FIXME: Parameters 'random_seed', 'site_class' unfilled
        ac = allel.AlleleCountsArray(
            self.snp_allele_counts(
                region=region,
                sample_sets=sample_sets,
                sample_query=sample_query,
                site_mask=None,
                cohort_size=cohort_size,
            )
        )
        an = ac.sum(axis=1)
        is_seg = ac.is_segregating()
        is_var = ac.is_variant()
        allelism = ac.allelism()

        debug("obtain SNP variants data")
        ds_sites = self.snp_variants(
            region=region,
        ).compute()

        debug("build a dataframe")
        pos = ds_sites["variant_position"].values
        alleles = ds_sites["variant_allele"].values.astype("U")
        cols = {
            "pos": pos,
            "allele_0": alleles[:, 0],
            "allele_1": alleles[:, 1],
            "allele_2": alleles[:, 2],
            "allele_3": alleles[:, 3],
            "ac_0": ac[:, 0],
            "ac_1": ac[:, 1],
            "ac_2": ac[:, 2],
            "ac_3": ac[:, 3],
            "an": an,
            "is_seg": is_seg,
            "is_var": is_var,
            "allelism": allelism,
        }

        for site_mask_id in self._site_mask_ids():
            cols[f"pass_{site_mask_id}"] = ds_sites[
                f"variant_filter_pass_{site_mask_id}"
            ].values

        data = pd.DataFrame(cols)

        debug("create figure")
        xwheel_zoom = bkmod.WheelZoomTool(dimensions="width", maintain_focus=False)
        pos = data["pos"].values
        x_min = pos[0]
        x_max = pos[-1]
        if x_range is None:
            x_range = bkmod.Range1d(x_min, x_max, bounds="auto")

        tooltips = [
            ("Position", "$x{0,0}"),
            (
                "Alleles",
                "@allele_0 (@ac_0), @allele_1 (@ac_1), @allele_2 (@ac_2), @allele_3 (@ac_3)",
            ),
            ("No. alleles", "@allelism"),
            ("Allele calls", "@an"),
        ]

        for site_mask_id in self._site_mask_ids():
            tooltips.append((f"Pass {site_mask_id}", f"@pass_{site_mask_id}"))

        fig = bkplt.figure(
            title="SNPs",
            tools=["xpan", "xzoom_in", "xzoom_out", xwheel_zoom, "reset"],
            active_scroll=xwheel_zoom,
            active_drag="xpan",
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            toolbar_location="above",
            x_range=x_range,
            y_range=(0.5, 2.5),
            tooltips=tooltips,
        )
        hover_tool = fig.select(type=bkmod.HoverTool)
        hover_tool.names = ["snps"]

        debug("plot gaps in the reference genome")
        seq = self.genome_sequence(region=region.contig).compute()
        is_n = (seq == b"N") | (seq == b"n")
        loc_n_start = ~is_n[:-1] & is_n[1:]
        loc_n_stop = is_n[:-1] & ~is_n[1:]
        n_starts = np.nonzero(loc_n_start)[0]
        n_stops = np.nonzero(loc_n_stop)[0]
        df_n_runs = pd.DataFrame(
            {"left": n_starts + 1.6, "right": n_stops + 1.4, "top": 2.5, "bottom": 0.5}
        )
        fig.quad(
            top="top",
            bottom="bottom",
            left="left",
            right="right",
            color="#cccccc",
            source=df_n_runs,
            name="gaps",
        )

        debug("plot SNPs")
        color_pass = bkpal.Colorblind6[3]
        color_fail = bkpal.Colorblind6[5]
        data["left"] = data["pos"] - 0.4
        data["right"] = data["pos"] + 0.4
        data["bottom"] = np.where(data["is_seg"], 1.6, 0.6)
        data["top"] = data["bottom"] + 0.8
        data["color"] = np.where(data[f"pass_{site_mask}"], color_pass, color_fail)
        fig.quad(
            top="top",
            bottom="bottom",
            left="left",
            right="right",
            color="color",
            source=data,
            name="snps",
        )
        # TODO add legend?

        debug("tidy plot")
        fig.yaxis.ticker = bkmod.FixedTicker(
            ticks=[1, 2],
        )
        fig.yaxis.major_label_overrides = {
            1: "Non-segregating",
            2: "Segregating",
        }
        fig.xaxis.axis_label = f"Contig {region.contig} position (bp)"
        fig.xaxis.ticker = bkmod.AdaptiveTicker(min_interval=1)
        fig.xaxis.minor_tick_line_color = None
        fig.xaxis[0].formatter = bkmod.NumeralTickFormatter(format="0,0")

        if show:
            bkplt.show(fig)

        return fig

    def snp_allele_frequencies(
        self,
        transcript,
        cohorts,
        sample_query=None,
        min_cohort_size=10,
        site_mask=None,
        sample_sets=None,
        drop_invariant=True,
        effects=True,
    ):
        """Compute per variant allele frequencies for a gene transcript.

        Parameters
        ----------
        transcript : str
            Gene transcript ID, e.g., "AGAP004707-RD".
        cohorts : str or dict
            If a string, gives the name of a predefined cohort set, e.g., one of
            {"admin1_month", "admin1_year", "admin2_month", "admin2_year"}.
            If a dict, should map cohort labels to sample queries, e.g.,
            `{"bf_2012_col": "country == 'Burkina Faso' and year == 2012 and
            taxon == 'coluzzii'"}`.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        min_cohort_size : int
            Minimum cohort size. Any cohorts below this size are omitted.
        site_mask : str, optional
            Site filters mask to apply.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        drop_invariant : bool, optional
            If True, variants with no alternate allele calls in any cohorts are
            dropped from the result.
        effects : bool, optional
            If True, add SNP effect columns.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of SNP frequencies, one row per variant.

        Notes
        -----
        Cohorts with fewer samples than min_cohort_size will be excluded from
        output.

        """
        debug = self._log.debug

        debug("check parameters")
        self._check_param_min_cohort_size(min_cohort_size)

        debug("access sample metadata")
        df_samples = self.sample_metadata(
            sample_sets=sample_sets, sample_query=sample_query
        )

        debug("setup initial dataframe of SNPs")
        region, df_snps = self._snp_df(transcript=transcript)

        debug("get genotypes")
        gt = self.snp_genotypes(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            field="GT",
        )

        debug("slice to feature location")
        with self._dask_progress(desc="Load SNP genotypes"):
            gt = gt.compute()

        debug("build coh dict")
        coh_dict = self._locate_cohorts(cohorts=cohorts, df_samples=df_samples)

        debug("count alleles")
        freq_cols = dict()
        cohorts_iterator = self._progress(
            coh_dict.items(), desc="Compute allele frequencies"
        )
        for coh, loc_coh in cohorts_iterator:
            n_samples = np.count_nonzero(loc_coh)
            debug(f"{coh}, {n_samples} samples")
            if n_samples >= min_cohort_size:
                gt_coh = np.compress(loc_coh, gt, axis=1)
                ac_coh = allel.GenotypeArray(gt_coh).count_alleles(max_allele=3)
                af_coh = ac_coh.to_frequencies()
                freq_cols["frq_" + coh] = af_coh[:, 1:].flatten()

        debug("build a dataframe with the frequency columns")
        df_freqs = pd.DataFrame(freq_cols)

        debug("compute max_af")
        df_max_af = pd.DataFrame({"max_af": df_freqs.max(axis=1)})

        debug("build the final dataframe")
        df_snps.reset_index(drop=True, inplace=True)
        df_snps = pd.concat([df_snps, df_freqs, df_max_af], axis=1)

        debug("apply site mask if requested")
        if site_mask is not None:
            loc_sites = df_snps[f"pass_{site_mask}"]
            df_snps = df_snps.loc[loc_sites]

        debug("drop invariants")
        if drop_invariant:
            loc_variant = df_snps["max_af"] > 0
            df_snps = df_snps.loc[loc_variant]

        debug("reset index after filtering")
        df_snps.reset_index(inplace=True, drop=True)

        if effects:

            debug("add effect annotations")
            ann = self._annotator()
            ann.get_effects(
                transcript=transcript, variants=df_snps, progress=self._progress
            )

            debug("add label")
            df_snps["label"] = self._pandas_apply(
                self._make_snp_label_effect,
                df_snps,
                columns=["contig", "position", "ref_allele", "alt_allele", "aa_change"],
            )

            debug("set index")
            df_snps.set_index(
                ["contig", "position", "ref_allele", "alt_allele", "aa_change"],
                inplace=True,
            )

        else:

            debug("add label")
            df_snps["label"] = self._pandas_apply(
                self._make_snp_label,
                df_snps,
                columns=["contig", "position", "ref_allele", "alt_allele"],
            )

            debug("set index")
            df_snps.set_index(
                ["contig", "position", "ref_allele", "alt_allele"],
                inplace=True,
            )

        debug("add dataframe metadata")
        gene_name = self._transcript_to_gene_name(transcript)
        title = transcript
        if gene_name:
            title += f" ({gene_name})"
        title += " SNP frequencies"
        df_snps.attrs["title"] = title

        return df_snps

    def _read_cohort_metadata(self, *, sample_set):
        """Read cohort metadata for a single sample set."""
        try:
            df = self._cache_cohort_metadata[sample_set]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            release_path = self._release_to_path(release)
            path_prefix = f"{self._base_path}/{release_path}/metadata"
            path = f"{path_prefix}/cohorts_{self._cohorts_analysis}/{sample_set}/samples.cohorts.csv"
            # N.B., not all cohort metadata files exist, need to handle FileNotFoundError
            try:
                with self._fs.open(path) as f:
                    df = pd.read_csv(f, na_values="")

                # ensure all column names are lower case
                df.columns = [c.lower() for c in df.columns]

                # rename some columns for consistent naming
                df.rename(
                    columns={
                        "adm1_iso": "admin1_iso",
                        "adm1_name": "admin1_name",
                        "adm2_name": "admin2_name",
                    },
                    inplace=True,
                )
            except FileNotFoundError:
                # Specify cohort_cols
                cohort_cols = (
                    "country_iso",
                    "admin1_name",
                    "admin1_iso",
                    "admin2_name",
                    "taxon",
                    "cohort_admin1_year",
                    "cohort_admin1_month",
                    "cohort_admin2_year",
                    "cohort_admin2_month",
                )

                # Get sample ids as an index via general metadata (has caching)
                df_general = self._read_general_metadata(sample_set=sample_set)
                df_general.set_index("sample_id", inplace=True)

                # Create a blank DataFrame with cohort_cols and sample_id index
                df = pd.DataFrame(columns=cohort_cols, index=df_general.index.copy())

                # Revert sample_id index to column
                df.reset_index(inplace=True)

            self._cache_cohort_metadata[sample_set] = df
        return df.copy()

    def sample_cohorts(self, sample_sets=None):
        """Access cohorts metadata for one or more sample sets.

        Parameters
        ----------
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of cohort metadata, one row per sample.

        """
        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)

        # concatenate multiple sample sets
        dfs = [self._read_cohort_metadata(sample_set=s) for s in sample_sets]
        df = pd.concat(dfs, axis=0, ignore_index=True)

        return df

    def aa_allele_frequencies(
        self,
        transcript,
        cohorts,
        sample_query=None,
        min_cohort_size=10,
        site_mask=None,
        sample_sets=None,
        drop_invariant=True,
    ):
        """Compute per amino acid allele frequencies for a gene transcript.

        Parameters
        ----------
        transcript : str
            Gene transcript ID, e.g., "AGAP004707-RA".
        cohorts : str or dict
            If a string, gives the name of a predefined cohort set, e.g., one of
            {"admin1_month", "admin1_year", "admin2_month", "admin2_year"}.
            If a dict, should map cohort labels to sample queries, e.g.,
            `{"bf_2012_col": "country == 'Burkina Faso' and year == 2012 and
            taxon == 'coluzzii'"}`.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        min_cohort_size : int
            Minimum cohort size, below which allele frequencies are not
            calculated for cohorts.
        site_mask : str, optional
            Site filters mask to apply.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        drop_invariant : bool, optional
            If True, variants with no alternate allele calls in any cohorts are
            dropped from the result.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of amino acid allele frequencies, one row per
            replacement.

        Notes
        -----
        Cohorts with fewer samples than min_cohort_size will be excluded from
        output.

        """
        debug = self._log.debug

        df_snps = self.snp_allele_frequencies(
            transcript=transcript,
            cohorts=cohorts,
            sample_query=sample_query,
            min_cohort_size=min_cohort_size,
            site_mask=site_mask,
            sample_sets=sample_sets,
            drop_invariant=drop_invariant,
            effects=True,
        )
        df_snps.reset_index(inplace=True)

        # we just want aa change
        df_ns_snps = df_snps.query(AA_CHANGE_QUERY).copy()

        # N.B., we need to worry about the possibility of the
        # same aa change due to SNPs at different positions. We cannot
        # sum frequencies of SNPs at different genomic positions. This
        # is why we group by position and aa_change, not just aa_change.

        debug("group and sum to collapse multi variant allele changes")
        freq_cols = [col for col in df_ns_snps if col.startswith("frq")]
        agg = {c: np.nansum for c in freq_cols}
        keep_cols = (
            "contig",
            "transcript",
            "aa_pos",
            "ref_allele",
            "ref_aa",
            "alt_aa",
            "effect",
            "impact",
        )
        for c in keep_cols:
            agg[c] = "first"
        agg["alt_allele"] = lambda v: "{" + ",".join(v) + "}" if len(v) > 1 else v
        df_aaf = df_ns_snps.groupby(["position", "aa_change"]).agg(agg).reset_index()

        debug("compute new max_af")
        df_aaf["max_af"] = df_aaf[freq_cols].max(axis=1)

        debug("add label")
        df_aaf["label"] = self._pandas_apply(
            self._make_snp_label_aa,
            df_aaf,
            columns=["aa_change", "contig", "position", "ref_allele", "alt_allele"],
        )

        debug("sort by genomic position")
        df_aaf = df_aaf.sort_values(["position", "aa_change"])

        debug("set index")
        df_aaf.set_index(["aa_change", "contig", "position"], inplace=True)

        debug("add metadata")
        gene_name = self._transcript_to_gene_name(transcript)
        title = transcript
        if gene_name:
            title += f" ({gene_name})"
        title += " SNP frequencies"
        df_aaf.attrs["title"] = title

        return df_aaf

    def aa_allele_frequencies_advanced(
        self,
        transcript,
        area_by,
        period_by,
        sample_sets=None,
        sample_query=None,
        min_cohort_size=10,
        variant_query=None,
        site_mask=None,
        nobs_mode="called",  # or "fixed"
        ci_method="wilson",
    ):
        """Group samples by taxon, area (space) and period (time), then compute
        amino acid change allele counts and frequencies.

        Parameters
        ----------
        transcript : str
            Gene transcript ID, e.g., "AGAP004707-RD".
        area_by : str
            Column name in the sample metadata to use to group samples spatially.
            E.g., use "admin1_iso" or "admin1_name" to group by level 1
            administrative divisions, or use "admin2_name" to group by level 2
            administrative divisions.
        period_by : {"year", "quarter", "month"}
            Length of time to group samples temporally.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        min_cohort_size : int, optional
            Minimum cohort size. Any cohorts below this size are omitted.
        variant_query : str, optional
        site_mask : str, optional
            Site filters mask to apply.
        nobs_mode : {"called", "fixed"}
            Method for calculating the denominator when computing frequencies.
            If "called" then use the number of called alleles, i.e., number of
            samples with non-missing genotype calls multiplied by 2. If "fixed"
            then use the number of samples multiplied by 2.
        ci_method : {"normal", "agresti_coull", "beta", "wilson", "binom_test"}, optional
            Method to use for computing confidence intervals, passed through to
            `statsmodels.stats.proportion.proportion_confint`.

        Returns
        -------
        ds : xarray.Dataset
            The resulting dataset contains data has dimensions "cohorts" and
            "variants". Variables prefixed with "cohort" are 1-dimensional
            arrays with data about the cohorts, such as the area, period, taxon
            and cohort size. Variables prefixed with "variant" are 1-dimensional
            arrays with data about the variants, such as the contig, position,
            reference and alternate alleles. Variables prefixed with "event" are
            2-dimensional arrays with the allele counts and frequency
            calculations.

        """
        debug = self._log.debug

        debug("begin by computing SNP allele frequencies")
        ds_snp_frq = self.snp_allele_frequencies_advanced(
            transcript=transcript,
            area_by=area_by,
            period_by=period_by,
            sample_sets=sample_sets,
            sample_query=sample_query,
            min_cohort_size=min_cohort_size,
            drop_invariant=True,  # always drop invariant for aa frequencies
            variant_query=AA_CHANGE_QUERY,  # we'll also apply a variant query later
            site_mask=site_mask,
            nobs_mode=nobs_mode,
            ci_method=None,  # we will recompute confidence intervals later
        )

        # N.B., we need to worry about the possibility of the
        # same aa change due to SNPs at different positions. We cannot
        # sum frequencies of SNPs at different genomic positions. This
        # is why we group by position and aa_change, not just aa_change.

        # add in a special grouping column to work around the fact that xarray currently
        # doesn't support grouping by multiple variables in the same dimension
        df_grouper = ds_snp_frq[
            ["variant_position", "variant_aa_change"]
        ].to_dataframe()
        grouper_var = df_grouper.apply(
            lambda row: "_".join([str(v) for v in row]), axis="columns"
        )
        ds_snp_frq["variant_position_aa_change"] = "variants", grouper_var

        debug("group by position and amino acid change")
        group_by_aa_change = ds_snp_frq.groupby("variant_position_aa_change")

        debug("apply aggregation")
        ds_aa_frq = group_by_aa_change.map(self._map_snp_to_aa_change_frq_ds)

        debug("add back in cohort variables, unaffected by aggregation")
        # FIXME: Unresolved attribute reference 'startswith' for class 'Hashable'
        cohort_vars = [v for v in ds_snp_frq if v.startswith("cohort_")]
        for v in cohort_vars:
            ds_aa_frq[v] = ds_snp_frq[v]

        debug("sort by genomic position")
        ds_aa_frq = ds_aa_frq.sortby(["variant_position", "variant_aa_change"])

        debug("recompute frequency")
        count = ds_aa_frq["event_count"].values
        nobs = ds_aa_frq["event_nobs"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            frequency = count / nobs  # ignore division warnings
        ds_aa_frq["event_frequency"] = ("variants", "cohorts"), frequency

        debug("recompute max frequency over cohorts")
        with warnings.catch_warnings():
            # ignore "All-NaN slice encountered" warnings
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_af = np.nanmax(ds_aa_frq["event_frequency"].values, axis=1)
        ds_aa_frq["variant_max_af"] = "variants", max_af

        debug("set up variant dataframe, useful intermediate")
        variant_cols = [v for v in ds_aa_frq if v.startswith("variant_")]
        df_variants = ds_aa_frq[variant_cols].to_dataframe()
        df_variants.columns = [c.split("variant_")[1] for c in df_variants.columns]

        debug("assign new variant label")
        label = self._pandas_apply(
            self._make_snp_label_aa,
            df_variants,
            columns=["aa_change", "contig", "position", "ref_allele", "alt_allele"],
        )
        ds_aa_frq["variant_label"] = "variants", label

        debug("apply variant query if given")
        if variant_query is not None:
            loc_variants = df_variants.eval(variant_query).values
            ds_aa_frq = ds_aa_frq.isel(variants=loc_variants)

        debug("compute new confidence intervals")
        self._add_frequency_ci(ds_aa_frq, ci_method)

        debug("tidy up display by sorting variables")
        ds_aa_frq = ds_aa_frq[sorted(ds_aa_frq)]

        gene_name = self._transcript_to_gene_name(transcript)
        title = transcript
        if gene_name:
            title += f" ({gene_name})"
        title += " SNP frequencies"
        ds_aa_frq.attrs["title"] = title

        return ds_aa_frq

    def _block_jackknife_cohort_diversity_stats(
        self, *, cohort_label, ac, n_jack, confidence_level
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
        # FIXME: variable in function should be lowercase
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
            # FIXME: variable in function should be lowercase
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
        ) = jackknife_ci(
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
        ) = jackknife_ci(
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
        ) = jackknife_ci(
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

    def cohort_diversity_stats(
        self,
        cohort,
        cohort_size,
        region,
        site_mask,
        site_class=None,
        sample_sets=None,
        random_seed=42,
        n_jack=200,
        confidence_level=0.95,
    ):
        """Compute genetic diversity summary statistics for a cohort of
        individuals.

        Parameters
        ----------
        cohort : str or (str, str)
            Either a string giving one of the predefined cohort labels, or a
            pair of strings giving a custom cohort label and a sample query to
            select samples in the cohort.
        cohort_size : int
            Number of individuals to use for computation of summary statistics.
            If the cohort is larger than this size, it will be randomly
            down-sampled.
        region : str
            Chromosome arm, gene name or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        site_mask : str
            Site filters mask to apply.
        site_class : str, optional
            Select sites belonging to one of the following classes: CDS_DEG_4,
            (4-fold degenerate coding sites), CDS_DEG_2_SIMPLE (2-fold simple
            degenerate coding sites), CDS_DEG_0 (non-degenerate coding sites),
            INTRON_SHORT (introns shorter than 100 bp), INTRON_LONG (introns
            longer than 200 bp), INTRON_SPLICE_5PRIME (intron within 2 bp of
            5' splice site), INTRON_SPLICE_3PRIME (intron within 2 bp of 3'
            splice site), UTR_5PRIME (5' untranslated region), UTR_3PRIME (3'
            untranslated region), INTERGENIC (intergenic, more than 10 kbp from
            a gene).
        sample_sets : str or list of str, optional
            Can be a sample set identifier or a list of
            sample set identifiers  or a
            release identifier or a list of release identifiers.
        random_seed : int, optional
            Seed for random number generator.
        n_jack : int, optional
            Number of blocks to divide the data into for the block jackknife
            estimation of confidence intervals. N.B., larger is not necessarily
            better.
        confidence_level : float, optional
            Confidence level to use for confidence interval calculation. 0.95
            means 95% confidence interval.

        Returns
        -------
        stats : pandas.Series
            A series with summary statistics and their confidence intervals.

        """

        debug = self._log.debug

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
            raise TypeError(r"invalid cohort parameter: {cohort!r}")

        debug("access allele counts")
        ac = self.snp_allele_counts(
            region=region,
            site_mask=site_mask,
            site_class=site_class,
            sample_query=cohort_query,
            sample_sets=sample_sets,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )

        debug("compute diversity stats")
        stats = self._block_jackknife_cohort_diversity_stats(
            cohort_label=cohort_label,
            ac=ac,
            n_jack=n_jack,
            confidence_level=confidence_level,
        )

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
                vals = df_samples[field].sort_values().unique()
                if len(vals) == 0:
                    val = np.nan
                elif len(vals) == 1:
                    val = vals[0]
                else:
                    val = vals.tolist()
            elif agg == "mean":
                vals = df_samples[field]
                if len(vals) == 0:
                    val = np.nan
                else:
                    val = np.mean(vals)
            else:
                val = np.nan
            stats[field] = val

        return pd.Series(stats)

    # TODO: compare with cohort_diversity_stats()
    def diversity_stats(
        self,
        cohorts,
        cohort_size,
        region,
        site_mask,
        site_class=None,
        sample_query=None,
        sample_sets=None,
        random_seed=42,
        n_jack=200,
        confidence_level=0.95,
    ):
        """Compute genetic diversity summary statistics for multiple cohorts.

        Parameters
        ----------
        cohorts : str or dict
            Either a string giving one of the predefined cohort columns, or a
            dictionary mapping cohort labels to sample queries.
        cohort_size : int
            Number of individuals to use for computation of summary statistics.
            If the cohort is larger than this size, it will be randomly
            down-sampled.
        region : str
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_class : str, optional
            Select sites belonging to one of the following classes: CDS_DEG_4,
            (4-fold degenerate coding sites), CDS_DEG_2_SIMPLE (2-fold simple
            degenerate coding sites), CDS_DEG_0 (non-degenerate coding sites),
            INTRON_SHORT (introns shorter than 100 bp), INTRON_LONG (introns
            longer than 200 bp), INTRON_SPLICE_5PRIME (intron within 2 bp of
            5' splice site), INTRON_SPLICE_3PRIME (intron within 2 bp of 3'
            splice site), UTR_5PRIME (5' untranslated region), UTR_3PRIME (3'
            untranslated region), INTERGENIC (intergenic, more than 10 kbp from
            a gene).
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        random_seed : int, optional
            Seed for random number generator.
        n_jack : int, optional
            Number of blocks to divide the data into for the block jackknife
            estimation of confidence intervals. N.B., larger is not necessarily
            better.
        confidence_level : float, optional
            Confidence level to use for confidence interval calculation. 0.95
            means 95% confidence interval.

        Returns
        -------
        df_stats : pandas.DataFrame
            A DataFrame where each row provides summary statistics and their
            confidence intervals for a single cohort.

        """
        debug = self._log.debug
        info = self._log.info

        debug("set up cohorts")
        if isinstance(cohorts, dict):
            # user has supplied a custom dictionary mapping cohort identifiers
            # to pandas queries
            cohort_queries = cohorts

        elif isinstance(cohorts, str):
            # user has supplied one of the predefined cohort sets

            df_samples = self.sample_metadata(
                sample_sets=sample_sets, sample_query=sample_query
            )

            # determine column in dataframe - allow abbreviation
            if cohorts.startswith("cohort_"):
                cohorts_col = cohorts
            else:
                cohorts_col = "cohort_" + cohorts
            if cohorts_col not in df_samples.columns:
                raise ValueError(f"{cohorts_col!r} is not a known cohort set")

            # find cohort labels and build queries dictionary
            cohort_labels = sorted(df_samples[cohorts_col].dropna().unique())
            cohort_queries = {coh: f"{cohorts_col} == '{coh}'" for coh in cohort_labels}

        else:
            raise TypeError("cohorts parameter should be dict or str")

        debug("handle sample_query parameter")
        if sample_query is not None:
            cohort_queries = {
                cohort_label: f"({cohort_query}) and ({sample_query})"
                for cohort_label, cohort_query in cohort_queries.items()
            }

        debug("check cohort sizes, drop any cohorts which are too small")
        cohort_queries_checked = dict()
        for cohort_label, cohort_query in cohort_queries.items():
            df_cohort_samples = self.sample_metadata(
                sample_sets=sample_sets, sample_query=cohort_query
            )
            n_samples = len(df_cohort_samples)
            if n_samples < cohort_size:
                info(
                    f"cohort ({cohort_label}) has insufficient samples ({n_samples}) for requested cohort size ({cohort_size}), dropping"  # noqa
                )  # noqa
            else:
                cohort_queries_checked[cohort_label] = cohort_query

        debug("compute diversity stats for cohorts")
        all_stats = []
        for cohort_label, cohort_query in cohort_queries_checked.items():
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
            )
            all_stats.append(stats)
        df_stats = pd.DataFrame(all_stats)

        return df_stats

    def count_samples(
        self,
        sample_sets=None,
        sample_query=None,
        index=(
            "country",
            "admin1_iso",
            "admin1_name",
            "admin2_name",
            "year",
        ),
        columns="taxon",
    ):
        """Create a pivot table showing numbers of samples available by space,
        time and taxon.

        Parameters
        ----------
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        index : str or tuple of str
            Sample metadata columns to use for the index.
        columns : str or tuple of str
            Sample metadata columns to use for the columns.

        Returns
        -------
        df : pandas.DataFrame
            Pivot table of sample counts.

        """
        debug = self._log.debug

        debug("load sample metadata")
        df_samples = self.sample_metadata(
            sample_sets=sample_sets, sample_query=sample_query
        )

        debug("create pivot table")
        df_pivot = df_samples.pivot_table(
            index=index,
            columns=columns,
            values="sample_id",
            aggfunc="count",
            fill_value=0,
        )

        return df_pivot

    def fst_gwss(
        self,
        contig,
        window_size,
        cohort1_query,
        cohort2_query,
        sample_sets=None,
        site_mask="default",
        cohort_size=30,
        random_seed=42,
    ):
        """Run a Fst genome-wide scan to investigate genetic differentiation
        between two cohorts.

        Parameters
        ----------
        contig: str
            Chromosome arm (e.g., "2L")
        window_size : int
            The size of windows used to calculate h12 over.
        cohort1_query : str
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        cohort2_query : str
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}, optional
            Site filters mask to apply.
        cohort_size : int, optional
            If provided, randomly down-sample to the given cohort size.
        random_seed : int, optional
            Random seed used for down-sampling.

        Returns
        -------
        x : numpy.ndarray
            An array containing the window centre point genomic positions.
        fst : numpy.ndarray
            An array with Fst statistic values for each window.
        """

        # change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data
        name = self._fst_gwss_results_cache_name

        if site_mask == "default":
            site_mask = self._default_site_mask

        params = dict(
            contig=contig,
            window_size=window_size,
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
            sample_sets=self._prep_sample_sets_arg(sample_sets=sample_sets),
            site_mask=site_mask,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )

        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._fst_gwss(**params)
            self.results_cache_set(name=name, params=params, results=results)

        x = results["x"]
        fst = results["fst"]

        return x, fst

    def _fst_gwss(
        self,
        contig,
        window_size,
        sample_sets,
        cohort1_query,
        cohort2_query,
        site_mask,
        cohort_size,
        random_seed,
    ):

        ds_snps1 = self.snp_calls(
            region=contig,
            sample_query=cohort1_query,
            sample_sets=sample_sets,
            site_mask=site_mask,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )

        ds_snps2 = self.snp_calls(
            region=contig,
            sample_query=cohort2_query,
            sample_sets=sample_sets,
            site_mask=site_mask,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )

        gt1 = allel.GenotypeDaskArray(ds_snps1["call_genotype"].data)
        with self._dask_progress(desc="Compute allele counts for cohort 1"):
            ac1 = gt1.count_alleles(max_allele=3).compute()

        gt2 = allel.GenotypeDaskArray(ds_snps2["call_genotype"].data)
        with self._dask_progress(desc="Compute allele counts for cohort 2"):
            ac2 = gt2.count_alleles(max_allele=3).compute()

        pos = ds_snps1["variant_position"].values

        fst = allel.moving_hudson_fst(ac1, ac2, size=window_size)
        x = allel.moving_statistic(pos, statistic=np.mean, size=window_size)

        results = dict(x=x, fst=fst)

        return results

    # Start of plot functions

    def plot_frequencies_heatmap(
        self,
        df,
        index="label",
        max_len=100,
        x_label="Cohorts",
        y_label="Variants",
        colorbar=True,
        col_width=40,
        width=None,
        row_height=20,
        height=None,
        text_auto=".0%",
        aspect="auto",
        color_continuous_scale="Reds",
        title=True,
        **kwargs,
    ):
        """Plot a heatmap from a pandas DataFrame of frequencies, e.g., output
        from `snp_allele_frequencies()` or `gene_cnv_frequencies()`.
        It's recommended to filter the input DataFrame to just rows of interest,
        i.e., fewer rows than `max_len`.

        Parameters
        ----------
        df : pandas DataFrame
           A DataFrame of frequencies, e.g., output from
           `snp_allele_frequencies()` or `gene_cnv_frequencies()`.
        index : str or list of str
            One or more column headers that are present in the input dataframe.
            This becomes the heatmap y-axis row labels. The column/s must
            produce a unique index.
        max_len : int, optional
            Displaying large styled dataframes may cause ipython notebooks to
            crash.
        x_label : str, optional
            This is the x-axis label that will be displayed on the heatmap.
        y_label : str, optional
            This is the y-axis label that will be displayed on the heatmap.
        colorbar : bool, optional
            If False, colorbar is not output.
        col_width : int, optional
            Plot width per column in pixels (px).
        width : int, optional
            Plot width in pixels (px), overrides col_width.
        row_height : int, optional
            Plot height per row in pixels (px).
        height : int, optional
            Plot height in pixels (px), overrides row_height.
        text_auto : str, optional
            Formatting for frequency values.
        aspect : str, optional
            Control the aspect ratio of the heatmap.
        color_continuous_scale : str, optional
            Color scale to use.
        title : bool or str, optional
            If True, attempt to use metadata from input dataset as a plot
            title. Otherwise, use supplied value as a title.
        **kwargs
            Other parameters are passed through to px.imshow().

        Returns
        -------
        fig : plotly.graph_objects.Figure

        """
        debug = self._log.debug

        import plotly.express as px

        debug("check len of input")
        if len(df) > max_len:
            raise ValueError(f"Input DataFrame is longer than {max_len}")

        debug("handle title")
        if title is True:
            title = df.attrs.get("title", None)

        debug("indexing")
        if index is None:
            index = list(df.index.names)
        df = df.reset_index().copy()
        if isinstance(index, list):
            index_col = (
                df[index]
                .astype(str)
                .apply(
                    lambda row: ", ".join([o for o in row if o is not None]),
                    axis="columns",
                )
            )
        elif isinstance(index, str):
            index_col = df[index].astype(str)
        else:
            raise TypeError("wrong type for index parameter, expected list or str")

        debug("check that index is unique")
        if not index_col.is_unique:
            raise ValueError(f"{index} does not produce a unique index")

        debug("drop and re-order columns")
        frq_cols = [col for col in df.columns if col.startswith("frq_")]

        debug("keep only freq cols")
        heatmap_df = df[frq_cols].copy()

        debug("set index")
        heatmap_df.set_index(index_col, inplace=True)

        debug("clean column names")
        heatmap_df.columns = heatmap_df.columns.str.lstrip("frq_")

        debug("deal with width and height")
        if width is None:
            width = 400 + col_width * len(heatmap_df.columns)
            if colorbar:
                width += 40
        if height is None:
            height = 200 + row_height * len(heatmap_df)
            if title is not None:
                height += 40

        debug("plotly heatmap styling")
        fig = px.imshow(
            img=heatmap_df,
            zmin=0,
            zmax=1,
            width=width,
            height=height,
            text_auto=text_auto,
            aspect=aspect,
            color_continuous_scale=color_continuous_scale,
            title=title,
            **kwargs,
        )

        fig.update_xaxes(side="bottom", tickangle=30)
        if x_label is not None:
            fig.update_xaxes(title=x_label)
        if y_label is not None:
            fig.update_yaxes(title=y_label)
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Frequency",
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["0%", "20%", "40%", "60%", "80%", "100%"],
            )
        )
        if not colorbar:
            fig.update(layout_coloraxis_showscale=False)

        return fig

    def plot_frequencies_time_series(
        self, ds, height=None, width=None, title=True, **kwargs
    ):
        """Create a time series plot of variant frequencies using plotly.

        Parameters
        ----------
        ds : xarray.Dataset
            A dataset of variant frequencies, such as returned by
            `snp_allele_frequencies_advanced()`,
            `aa_allele_frequencies_advanced()` or
            `gene_cnv_frequencies_advanced()`.
        height : int, optional
            Height of plot in pixels (px).
        width : int, optional
            Width of plot in pixels (px).
        title : bool or str, optional
            If True, attempt to use metadata from input dataset as a plot
            title. Otherwise, use supplied value as a title.
        **kwargs
            Passed through to `px.line()`.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            A plotly figure containing line graphs. The resulting figure will
            have one panel per cohort, grouped into columns by taxon, and
            grouped into rows by area. Markers and lines show frequencies of
            variants.

        """
        debug = self._log.debug

        import plotly.express as px

        debug("handle title")
        if title is True:
            title = ds.attrs.get("title", None)

        debug("extract cohorts into a dataframe")
        # FIXME: unresolved attribute reference 'startswith'
        cohort_vars = [v for v in ds if v.startswith("cohort_")]
        df_cohorts = ds[cohort_vars].to_dataframe()
        df_cohorts.columns = [c.split("cohort_")[1] for c in df_cohorts.columns]

        debug("extract variant labels")
        variant_labels = ds["variant_label"].values

        debug("build a long-form dataframe from the dataset")
        dfs = []
        for cohort_index, cohort in enumerate(df_cohorts.itertuples()):
            ds_cohort = ds.isel(cohorts=cohort_index)
            df = pd.DataFrame(
                {
                    "taxon": cohort.taxon,
                    "area": cohort.area,
                    "date": cohort.period_start,
                    "period": str(
                        cohort.period
                    ),  # use string representation for hover label
                    "sample_size": cohort.size,
                    "variant": variant_labels,
                    "count": ds_cohort["event_count"].values,
                    "nobs": ds_cohort["event_nobs"].values,
                    "frequency": ds_cohort["event_frequency"].values,
                    "frequency_ci_low": ds_cohort["event_frequency_ci_low"].values,
                    "frequency_ci_upp": ds_cohort["event_frequency_ci_upp"].values,
                }
            )
            dfs.append(df)
        df_events = pd.concat(dfs, axis=0).reset_index(drop=True)

        debug("remove events with no observations")
        df_events = df_events.query("nobs > 0")

        debug("calculate error bars")
        frq = df_events["frequency"]
        frq_ci_low = df_events["frequency_ci_low"]
        frq_ci_upp = df_events["frequency_ci_upp"]
        df_events["frequency_error"] = frq_ci_upp - frq
        df_events["frequency_error_minus"] = frq - frq_ci_low

        debug("make a plot")
        fig = px.line(
            df_events,
            facet_col="taxon",
            facet_row="area",
            x="date",
            y="frequency",
            error_y="frequency_error",
            error_y_minus="frequency_error_minus",
            color="variant",
            markers=True,
            hover_name="variant",
            hover_data={
                "frequency": ":.0%",
                "period": True,
                "area": True,
                "taxon": True,
                "sample_size": True,
                "date": False,
                "variant": False,
            },
            height=height,
            width=width,
            title=title,
            labels={
                "date": "Date",
                "frequency": "Frequency",
                "variant": "Variant",
                "taxon": "Taxon",
                "area": "Area",
                "period": "Period",
                "sample_size": "Sample size",
            },
            **kwargs,
        )

        debug("tidy plot")
        fig.update_layout(yaxis_range=[-0.05, 1.05])

        return fig

    def plot_frequencies_map_markers(self, m, ds, variant, taxon, period, clear=True):
        """Plot markers on a map showing variant frequencies for cohorts grouped
        by area (space), period (time) and taxon.

        Parameters
        ----------
        m : ipyleaflet.Map
            The map on which to add the markers.
        ds : xarray.Dataset
            A dataset of variant frequencies, such as returned by
            `snp_allele_frequencies_advanced()`,
            `aa_allele_frequencies_advanced()` or
            `gene_cnv_frequencies_advanced()`.
        variant : int or str
            Index or label of variant to plot.
        taxon : str
            Taxon to show markers for.
        period : pd.Period
            Time period to show markers for.
        clear : bool, optional
            If True, clear all layers (except the base layer) from the map
            before adding new markers.

        """
        debug = self._log.debug

        import ipyleaflet
        import ipywidgets

        debug("slice dataset to variant of interest")
        if isinstance(variant, int):
            ds_variant = ds.isel(variants=variant)
            variant_label = ds["variant_label"].values[variant]
        elif isinstance(variant, str):
            ds_variant = ds.set_index(variants="variant_label").sel(variants=variant)
            variant_label = variant
        else:
            raise TypeError(
                f"Bad type for variant parameter; expected int or str, found {type(variant)}."
            )

        debug("convert to a dataframe for convenience")
        df_markers = ds_variant[
            [
                "cohort_taxon",
                "cohort_area",
                "cohort_period",
                "cohort_lat_mean",
                "cohort_lon_mean",
                "cohort_size",
                "event_frequency",
                "event_frequency_ci_low",
                "event_frequency_ci_upp",
            ]
        ].to_dataframe()

        debug("select data matching taxon and period parameters")
        df_markers = df_markers.loc[
            (
                (df_markers["cohort_taxon"] == taxon)
                & (df_markers["cohort_period"] == period)
            )
        ]

        debug("clear existing layers in the map")
        if clear:
            for layer in m.layers[1:]:
                m.remove(layer)

        debug("add markers")
        for x in df_markers.itertuples():
            marker = ipyleaflet.CircleMarker()
            marker.location = (x.cohort_lat_mean, x.cohort_lon_mean)
            marker.radius = 20
            marker.color = "black"
            marker.weight = 1
            marker.fill_color = "red"
            marker.fill_opacity = x.event_frequency
            popup_html = f"""
                <strong>{variant_label}</strong> <br/>
                Taxon: {x.cohort_taxon} <br/>
                Area: {x.cohort_area} <br/>
                Period: {x.cohort_period} <br/>
                Sample size: {x.cohort_size} <br/>
                Frequency: {x.event_frequency:.0%}
                (95% CI: {x.event_frequency_ci_low:.0%} - {x.event_frequency_ci_upp:.0%})
            """
            marker.popup = ipyleaflet.Popup(
                child=ipywidgets.HTML(popup_html),
            )
            m.add(marker)

    def plot_frequencies_interactive_map(
        self,
        ds,
        center=(-2, 20),
        zoom=3,
        title=True,
        epilogue=True,
    ):
        """Create an interactive map with markers showing variant frequencies or
        cohorts grouped by area (space), period (time) and taxon.

        Parameters
        ----------
        ds : xarray.Dataset
            A dataset of variant frequencies, such as returned by
            `snp_allele_frequencies_advanced()`,
            `aa_allele_frequencies_advanced()` or
            `gene_cnv_frequencies_advanced()`.
        center : tuple of int, optional
            Location to center the map.
        zoom : int, optional
            Initial zoom level.
        title : bool or str, optional
            If True, attempt to use metadata from input dataset as a plot
            title. Otherwise, use supplied value as a title.
        epilogue : bool or str, optional
            Additional text to display below the map.

        Returns
        -------
        out : ipywidgets.Widget
            An interactive map with widgets for selecting which variant, taxon
            and time period to display.

        """
        debug = self._log.debug

        import ipyleaflet
        import ipywidgets

        debug("handle title")
        if title is True:
            title = ds.attrs.get("title", None)

        debug("create a map")
        freq_map = ipyleaflet.Map(center=center, zoom=zoom)

        debug("set up interactive controls")
        variants = ds["variant_label"].values
        taxa = np.unique(ds["cohort_taxon"].values)
        periods = np.unique(ds["cohort_period"].values)
        controls = ipywidgets.interactive(
            self.plot_frequencies_map_markers,
            m=ipywidgets.fixed(freq_map),
            ds=ipywidgets.fixed(ds),
            variant=ipywidgets.Dropdown(options=variants, description="Variant: "),
            taxon=ipywidgets.Dropdown(options=taxa, description="Taxon: "),
            period=ipywidgets.Dropdown(options=periods, description="Period: "),
            clear=ipywidgets.fixed(True),
        )

        debug("lay out widgets")
        components = []
        if title is not None:
            components.append(ipywidgets.HTML(value=f"<h3>{title}</h3>"))
        components.append(controls)
        components.append(freq_map)
        if epilogue is True:
            epilogue = """
                Variant frequencies are shown as coloured markers. Opacity of color
                denotes frequency. Click on a marker for more information.
            """
        if epilogue:
            components.append(ipywidgets.HTML(value=f"{epilogue}"))

        out = ipywidgets.VBox(components)

        return out

    def plot_pca_coords(
        self,
        data,
        x="PC1",
        y="PC2",
        color=None,
        symbol=None,
        jitter_frac=0.02,
        random_seed=42,
        width=900,
        height=600,
        marker_size=10,
        **kwargs,
    ):
        """Plot sample coordinates from a principal components analysis (PCA)
        as a plotly scatter plot.

        Parameters
        ----------
        data : pandas.DataFrame
            A dataframe of sample metadata, with columns "PC1", "PC2", "PC3",
            etc., added.
        x : str, optional
            Name of principal component to plot on the X axis.
        y : str, optional
            Name of principal component to plot on the Y axis.
        color : str, optional
            Name of column in the input dataframe to use to color the markers.
        symbol : str, optional
            Name of column in the input dataframe to use to choose marker symbols.
        jitter_frac : float, optional
            Randomly jitter points by this fraction of their range.
        random_seed : int, optional
            Random seed for jitter.
        width : int, optional
            Plot width in pixels (px).
        height : int, optional
            Plot height in pixels (px).
        marker_size : int, optional
            Marker size.

        Returns
        -------
        fig : Figure
            A plotly figure.

        """
        debug = self._log.debug

        import plotly.express as px

        debug(
            "set up data - copy and shuffle so that we don't get systematic over-plotting"
        )
        # TODO does the shuffling actually work?
        data = (
            data.copy().sample(frac=1, random_state=random_seed).reset_index(drop=True)
        )

        debug(
            "apply jitter if desired - helps spread out points when tightly clustered"
        )
        if jitter_frac:
            np.random.seed(random_seed)
            data[x] = jitter(data[x], jitter_frac)
            data[y] = jitter(data[y], jitter_frac)

        debug("convenience variables")
        data["country_location"] = data["country"] + " - " + data["location"]

        debug("set up plotting options")
        hover_data = [
            "partner_sample_id",
            "sample_set",
            "taxon",
            "country",
            "admin1_iso",
            "admin1_name",
            "admin2_name",
            "location",
            "year",
            "month",
        ]
        plot_kwargs = dict(
            width=width,
            height=height,
            color=color,
            symbol=symbol,
            template="simple_white",
            hover_name="sample_id",
            hover_data=hover_data,
            opacity=0.9,
            render_mode="svg",
        )

        debug("special handling for taxon color")
        if color == "taxon":
            self._setup_taxon_colors(plot_kwargs)

        debug("apply any user overrides")
        plot_kwargs.update(kwargs)

        debug("2D scatter plot")
        fig = px.scatter(data, x=x, y=y, **plot_kwargs)

        debug("tidy up")
        fig.update_layout(
            legend=dict(itemsizing="constant"),
        )
        fig.update_traces(marker={"size": marker_size})

        return fig

    def plot_pca_coords_3d(
        self,
        data,
        x="PC1",
        y="PC2",
        z="PC3",
        color=None,
        symbol=None,
        jitter_frac=0.02,
        random_seed=42,
        width=900,
        height=600,
        marker_size=5,
        **kwargs,
    ):
        """Plot sample coordinates from a principal components analysis (PCA)
        as a plotly 3D scatter plot.

        Parameters
        ----------
        data : pandas.DataFrame
            A dataframe of sample metadata, with columns "PC1", "PC2", "PC3",
            etc., added.
        x : str, optional
            Name of principal component to plot on the X axis.
        y : str, optional
            Name of principal component to plot on the Y axis.
        z : str, optional
            Name of principal component to plot on the Z axis.
        color : str, optional
            Name of column in the input dataframe to use to color the markers.
        symbol : str, optional
            Name of column in the input dataframe to use to choose marker symbols.
        jitter_frac : float, optional
            Randomly jitter points by this fraction of their range.
        random_seed : int, optional
            Random seed for jitter.
        width : int, optional
            Plot width in pixels (px).
        height : int, optional
            Plot height in pixels (px).
        marker_size : int, optional
            Marker size.

        Returns
        -------
        fig : Figure
            A plotly figure.

        """
        debug = self._log.debug

        import plotly.express as px

        debug(
            "set up data - copy and shuffle so that we don't get systematic over-plotting"
        )
        # TODO does this actually work?
        data = (
            data.copy().sample(frac=1, random_state=random_seed).reset_index(drop=True)
        )

        debug(
            "apply jitter if desired - helps spread out points when tightly clustered"
        )
        if jitter_frac:
            np.random.seed(random_seed)
            data[x] = jitter(data[x], jitter_frac)
            data[y] = jitter(data[y], jitter_frac)
            data[z] = jitter(data[z], jitter_frac)

        debug("convenience variables")
        data["country_location"] = data["country"] + " - " + data["location"]

        debug("set up plotting options")
        hover_data = [
            "partner_sample_id",
            "sample_set",
            "taxon",
            "country",
            "admin1_iso",
            "admin1_name",
            "admin2_name",
            "location",
            "year",
            "month",
        ]
        plot_kwargs = dict(
            width=width,
            height=height,
            hover_name="sample_id",
            hover_data=hover_data,
            color=color,
            symbol=symbol,
        )

        debug("special handling for taxon color")
        if color == "taxon":
            self._setup_taxon_colors(plot_kwargs)

        debug("apply any user overrides")
        plot_kwargs.update(kwargs)

        debug("3D scatter plot")
        fig = px.scatter_3d(data, x=x, y=y, z=z, **plot_kwargs)

        debug("tidy up")
        fig.update_layout(
            scene=dict(aspectmode="cube"),
            legend=dict(itemsizing="constant"),
        )
        fig.update_traces(marker={"size": marker_size})

        return fig

    def plot_diversity_stats(
        self,
        df_stats,
        color=None,
        bar_plot_height=450,
        bar_width=30,
        scatter_plot_height=500,
        scatter_plot_width=500,
        template="plotly_white",
        plot_kwargs=None,
    ):
        """Plot diversity statistics.

        Parameters
        ----------
        df_stats : pandas.DataFrame
            Output from diversity_stats().
        color : str, optional
            Column to color by.
        bar_plot_height : int, optional
            Height of bar plots in pixels (px).
        bar_width : int, optional
            Width per bar in pixels (px).
        scatter_plot_height : int, optional
            Height of scatter plot in pixels (px).
        scatter_plot_width : int, optional
            Width of scatter plot in pixels (px).
        template : str, optional
            Plotly template.
        plot_kwargs : dict, optional
            Extra plotting parameters

        """
        debug = self._log.debug
        import plotly.express as px

        debug("set up common plotting parameters")
        if plot_kwargs is None:
            plot_kwargs = dict()
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
                "theta_pi_estimate": r"$\widehat{\theta}_{\pi}$",
                "theta_w_estimate": r"$\widehat{\theta}_{w}$",
                "tajima_d_estimate": r"$D$",
                "cohort": "Cohort",
                "taxon": "Taxon",
                "country": "Country",
            },
        )
        if color == "taxon":
            self._setup_taxon_colors(plot_kwargs=default_plot_kwargs)
        default_plot_kwargs.update(plot_kwargs)
        plot_kwargs = default_plot_kwargs
        bar_plot_width = 300 + bar_width * len(df_stats)

        debug("nucleotide diversity bar plot")
        fig = px.bar(
            data_frame=df_stats,
            x="cohort",
            y="theta_pi_estimate",
            error_y="theta_pi_ci_err",
            title="Nucleotide diversity",
            color=color,
            height=bar_plot_height,
            width=bar_plot_width,
            template=template,
            **plot_kwargs,
        )
        fig.show()

        debug("Watterson's estimator bar plot")
        fig = px.bar(
            data_frame=df_stats,
            x="cohort",
            y="theta_w_estimate",
            error_y="theta_w_ci_err",
            title="Watterson's estimator",
            color=color,
            height=bar_plot_height,
            width=bar_plot_width,
            template=template,
            **plot_kwargs,
        )
        fig.show()

        debug("Tajima's D bar plot")
        fig = px.bar(
            data_frame=df_stats,
            x="cohort",
            y="tajima_d_estimate",
            error_y="tajima_d_ci_err",
            title="Tajima's D",
            color=color,
            height=bar_plot_height,
            width=bar_plot_width,
            template=template,
            **plot_kwargs,
        )
        fig.show()

        debug("scatter plot comparing diversity estimators")
        fig = px.scatter(
            data_frame=df_stats,
            x="theta_pi_estimate",
            y="theta_w_estimate",
            error_x="theta_pi_ci_err",
            error_y="theta_w_ci_err",
            title="Diversity estimators",
            color=color,
            width=scatter_plot_width,
            height=scatter_plot_height,
            template=template,
            **plot_kwargs,
        )
        fig.show()

    def plot_samples_interactive_map(
        self,
        sample_sets=None,
        sample_query=None,
        basemap=None,
        center=(-2, 20),
        zoom=3,
        min_samples=1,
    ):
        """Plot an interactive map showing sampling locations using ipyleaflet.

        Parameters
        ----------
        sample_sets : str or list of str, optional
            Can be a sample set identifier or a list of
            sample set identifiers or a
            release identifier or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        basemap : dict
            Basemap description coming from ipyleaflet.basemaps.
        center : tuple of int, optional
            Location to center the map.
        zoom : int, optional
            Initial zoom level.
        min_samples : int, optional
            Minimum number of samples required to show a marker for a given
            location.

        Returns
        -------
        samples_map : ipyleaflet.Map
            Ipyleaflet map widget.

        """
        debug = self._log.debug

        import ipyleaflet

        debug("load sample metadata")
        df_samples = self.sample_metadata(
            sample_sets=sample_sets, sample_query=sample_query
        )

        debug("compute locations")
        pivot_location_taxon = df_samples.pivot_table(
            index=[
                "country",
                "admin1_iso",
                "admin1_name",
                "admin2_name",
                "location",
                "latitude",
                "longitude",
            ],
            columns=["taxon"],
            values="sample_id",
            aggfunc="count",
            fill_value=0,
        )

        debug("create a map")
        if basemap is None:
            # FIXME: cannot find reference 'Esri'
            basemap = ipyleaflet.basemaps.Esri.WorldImagery
        samples_map = ipyleaflet.Map(
            center=center,
            zoom=zoom,
            basemap=basemap,
        )
        samples_map.add(ipyleaflet.ScaleControl(position="bottomleft"))
        # make the map a bit taller than the default
        samples_map.layout.height = "500px"

        debug("add markers")
        taxa = df_samples["taxon"].dropna().sort_values().unique()
        for _, row in pivot_location_taxon.reset_index().iterrows():
            title = (
                f"Location: {row.location} ({row.latitude:.3f}, {row.longitude:.3f})"
            )
            title += f"\nAdmin level 2: {row.admin2_name}"
            title += f"\nAdmin level 1: {row.admin1_name} ({row.admin1_iso})"
            title += f"\nCountry: {row.country}"
            title += "\nNo. specimens: "
            all_n = 0
            for taxon in taxa:
                n = row[taxon]
                all_n += n
                if n > 0:
                    title += f"{n} {taxon}; "
            if all_n >= min_samples:
                marker = ipyleaflet.Marker(
                    location=(row.latitude, row.longitude),
                    draggable=False,
                    title=title,
                )
                samples_map.add(marker)

        return samples_map

    def plot_fst_gwss_track(
        self,
        contig,
        window_size,
        cohort1_query,
        cohort2_query,
        sample_sets=None,
        site_mask="default",
        cohort_size=30,
        random_seed=42,
        title=None,
        sizing_mode=DEFAULT_GENOME_PLOT_SIZING_MODE,
        width=DEFAULT_GENOME_PLOT_WIDTH,
        height=200,
        show=True,
        x_range=None,
    ):
        """Run and plot a Fst genome-wide scan to investigate genetic
            differentiation between two cohorts.
        Parameters
        ----------
        contig: str
            Contig name (e.g., "X")
        window_size : int
            The size of windows used to calculate h12 over.
        cohort1_query : str
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        cohort2_query : str
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        sample_sets : str or list of str, optional
            Can be a sample set identifier or a list of
            sample set identifiers or a
            release identifier or a list of release identifiers.
        site_mask : str, optional
            Site filters mask to apply.
        cohort_size : int, optional
            If provided, randomly down-sample to the given cohort size.
        random_seed : int, optional
            Random seed used for down-sampling.
        title : str, optional
            If provided, title string is used to label plot.
        sizing_mode : str, optional
            Bokeh plot sizing mode, see https://docs.bokeh.org/en/latest/docs/user_guide/layout.html#sizing-modes
        width : int, optional
            Plot width in pixels (px).
        height : int. optional
            Plot height in pixels (px).
        show : bool, optional
            If True, show the plot.
        x_range : bokeh.models.Range1d, optional
            X axis range (for linking to other tracks).

        Returns
        -------
        fig : figure
            A plot showing windowed Fst statistic across chosen contig.
        """

        import bokeh.models as bkmod
        import bokeh.plotting as bkplt

        # compute Fst
        x, fst = self.fst_gwss(
            contig=contig,
            window_size=window_size,
            cohort_size=cohort_size,
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
            sample_sets=sample_sets,
            site_mask=site_mask,
            random_seed=random_seed,
        )

        # determine X axis range
        x_min = x[0]
        x_max = x[-1]
        if x_range is None:
            x_range = bkmod.Range1d(x_min, x_max, bounds="auto")

        # create a figure
        xwheel_zoom = bkmod.WheelZoomTool(dimensions="width", maintain_focus=False)
        if title is None:
            title = f"Cohort 1: {cohort1_query}\nCohort 2: {cohort2_query}"
        fig = bkplt.figure(
            title=title,
            tools=["xpan", "xzoom_in", "xzoom_out", xwheel_zoom, "reset"],
            active_scroll=xwheel_zoom,
            active_drag="xpan",
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            toolbar_location="above",
            x_range=x_range,
            y_range=(0, 1),
        )

        # plot Fst
        fig.circle(
            x=x,
            y=fst,
            size=3,
            line_width=0.5,
            line_color="black",
            fill_color=None,
        )

        # tidy up the plot
        fig.yaxis.axis_label = "Fst"
        fig.yaxis.ticker = [0, 1]
        self._bokeh_style_genome_xaxis(fig, contig)

        if show:
            bkplt.show(fig)

        return fig

    def plot_fst_gwss(
        self,
        contig,
        window_size,
        cohort1_query,
        cohort2_query,
        sample_sets=None,
        site_mask="default",
        cohort_size=30,
        random_seed=42,
        title=None,
        sizing_mode=DEFAULT_GENOME_PLOT_SIZING_MODE,
        width=DEFAULT_GENOME_PLOT_WIDTH,
        track_height=190,
        genes_height=DEFAULT_GENES_TRACK_HEIGHT,
    ):
        """Run and plot a Fst genome-wide scan to investigate genetic
        differentiation between two cohorts.

        Parameters
        ----------
        contig: str
            Contig name (e.g., "X")
        window_size : int
            The size of windows used to calculate h12 over.
        cohort1_query : str
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        cohort2_query : str
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        sample_sets : str or list of str, optional
            Can be a sample set identifier or a list of
            sample set identifiers or a
            release identifier or a list of release identifiers.
        site_mask : str, optional
            Site filters mask to apply.
        cohort_size : int, optional
            If provided, randomly down-sample to the given cohort size.
        random_seed : int, optional
            Random seed used for down-sampling.
        title : str, optional
            If provided, title string is used to label plot.
        sizing_mode : str, optional
            Bokeh plot sizing mode, see https://docs.bokeh.org/en/latest/docs/user_guide/layout.html#sizing-modes
        width : int, optional
            Plot width in pixels (px).
        track_height : int. optional
            GWSS track height in pixels (px).
        genes_height : int. optional
            Gene track height in pixels (px).

        Returns
        -------
        fig : figure
            A plot showing windowed Fst statistic with gene track on x-axis.
        """

        import bokeh.layouts as bklay
        import bokeh.plotting as bkplt

        # gwss track
        fig1 = self.plot_fst_gwss_track(
            contig=contig,
            window_size=window_size,
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
            sample_sets=sample_sets,
            site_mask=site_mask,
            cohort_size=cohort_size,
            random_seed=random_seed,
            title=title,
            sizing_mode=sizing_mode,
            width=width,
            height=track_height,
            show=False,
        )

        fig1.xaxis.visible = False

        # plot genes
        fig2 = self.plot_genes(
            region=contig,
            sizing_mode=sizing_mode,
            width=width,
            height=genes_height,
            x_range=fig1.x_range,
            show=False,
        )

        # combine plots into a single figure
        fig = bklay.gridplot(
            [fig1, fig2],
            ncols=1,
            toolbar_location="above",
            merge_tools=True,
            sizing_mode=sizing_mode,
        )

        bkplt.show(fig)
