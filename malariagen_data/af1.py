import json
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from .anopheles import AnophelesDataResource

try:
    # noinspection PyPackageRequirements
    from google import colab
except ImportError:
    colab = None

import malariagen_data  # used for .__version__

from .util import CacheMiss, hash_params

MAJOR_VERSION_INT = 1
MAJOR_VERSION_GCS_STR = "v1.0"

GCS_URL = "gs://vo_afun_release/"

GENOME_FASTA_PATH = (
    "reference/genome/idAnoFuneDA-416_04/idAnoFuneDA-416_04_1.curated_primary.fa"
)
GENOME_FAI_PATH = (
    "reference/genome/idAnoFuneDA-416_04/idAnoFuneDA-416_04_1.curated_primary.fa.fai"
)
GENOME_ZARR_PATH = (
    "reference/genome/idAnoFuneDA-416_04/idAnoFuneDA-416_04_1.curated_primary.zarr"
)
SITE_ANNOTATIONS_ZARR_PATH = "reference/genome/idAnoFuneDA-416_04/Anopheles-funestus-DA-416_04_1_SEQANNOTATION.zarr"
GENOME_REF_ID = "idAnoFuneDA-416_04"
GENOME_REF_NAME = "Anopheles funestus"

CONTIGS = "2RL", "3RL", "X"

DEFAULT_GENOME_PLOT_WIDTH = 800  # width in px for bokeh genome plots
DEFAULT_GENES_TRACK_HEIGHT = 120  # height in px for bokeh genes track plots

PCA_RESULTS_CACHE_NAME = "af1_pca_v1"
SNP_ALLELE_COUNTS_CACHE_NAME = "af1_snp_allele_counts_v2"
DEFAULT_SITE_MASK = "funestus"


class Af1(AnophelesDataResource):
    """Provides access to data from Af1.x releases.

    Parameters
    ----------
    url : str
        Base path to data. Give "gs://vo_afun_release/" to use Google Cloud
        Storage, or a local path on your file system if data have been
        downloaded.
    site_filters_analysis : str, optional
        Site filters analysis version.
    bokeh_output_notebook : bool, optional
        If True (default), configure bokeh to output plots to the notebook.
    results_cache : str, optional
        Path to directory on local file system to save results.
    log : str or stream, optional
        File path or stream output for logging messages.
    debug : bool, optional
        Set to True to enable debug level logging.
    show_progress : bool, optional
        If True, show a progress bar during longer-running computations.
    check_location : bool, optional
        If True, use ipinfo to check the location of the client system.
    **kwargs
        Passed through to fsspec when setting up file system access.

    Examples
    --------
    Access data from Google Cloud Storage (default):

        >>> import malariagen_data
        >>> af1 = malariagen_data.Af1()

    Access data downloaded to a local file system:

        >>> af1 = malariagen_data.Af1("/local/path/to/vo_afun_release/")

    Access data from Google Cloud Storage, with caching on the local file system
    in a directory named "gcs_cache":

        >>> af1 = malariagen_data.Af1(
        ...     "simplecache::gs://vo_afun_release",
        ...     simplecache=dict(cache_storage="gcs_cache"),
        ... )

    Set up caching of some longer-running computations on the local file system,
    in a directory named "results_cache":

        >>> af1 = malariagen_data.Af1(results_cache="results_cache")

    """

    contigs = CONTIGS
    _major_version_int = MAJOR_VERSION_INT
    _major_version_gcs_str = MAJOR_VERSION_GCS_STR
    _genome_fasta_path = GENOME_FASTA_PATH
    _genome_fai_path = GENOME_FAI_PATH
    _genome_zarr_path = GENOME_ZARR_PATH
    _genome_ref_id = GENOME_REF_ID
    _genome_ref_name = GENOME_REF_NAME
    _gcs_url = GCS_URL
    _pca_results_cache_name = PCA_RESULTS_CACHE_NAME
    _default_site_mask = DEFAULT_SITE_MASK
    _site_annotations_zarr_path = SITE_ANNOTATIONS_ZARR_PATH
    _cohorts_analysis = None
    _site_filters_analysis = None

    def __init__(
        self,
        url=GCS_URL,
        bokeh_output_notebook=True,
        results_cache=None,
        log=sys.stdout,
        debug=False,
        show_progress=True,
        check_location=True,
        cohorts_analysis=None,
        site_filters_analysis=None,
        pre=False,
        **kwargs,  # used by simplecache, init_filesystem(url, **kwargs)
    ):

        super().__init__(
            url=url,
            site_filters_analysis=site_filters_analysis,
            bokeh_output_notebook=bokeh_output_notebook,
            results_cache=results_cache,
            log=log,
            debug=debug,
            show_progress=show_progress,
            check_location=check_location,
            pre=pre,
            **kwargs,  # used by simplecache, init_filesystem(url, **kwargs)
        )

        # load config.json
        path = f"{self._base_path}/v1.0-config.json"
        with self._fs.open(path) as f:
            self._config = json.load(f)

        if cohorts_analysis is None:
            self._cohorts_analysis = self._config["DEFAULT_COHORTS_ANALYSIS"]
        else:
            self._cohorts_analysis = cohorts_analysis

        if site_filters_analysis is None:
            self._site_filters_analysis = self._config["DEFAULT_SITE_FILTERS_ANALYSIS"]
        else:
            self._site_filters_analysis = site_filters_analysis

    @property
    def _public_releases(self):
        return tuple(self._config["PUBLIC_RELEASES"])

    @property
    def _geneset_gff3_path(self):
        return self._config["GENESET_GFF3_PATH"]

    @staticmethod
    def _setup_taxon_colors(plot_kwargs=None):
        import plotly.express as px

        if plot_kwargs is None:
            plot_kwargs = dict()
        taxon_palette = px.colors.qualitative.Plotly
        taxon_color_map = {
            "funestus": taxon_palette[0],
        }
        plot_kwargs.setdefault("color_discrete_map", taxon_color_map)
        plot_kwargs.setdefault(
            "category_orders", {"taxon": list(taxon_color_map.keys())}
        )
        return plot_kwargs

    def __repr__(self):
        text = (
            f"<MalariaGEN Af1 API client>\n"
            f"Storage URL             : {self._url}\n"
            f"Data releases available : {', '.join(self.releases)}\n"
            f"Results cache           : {self._results_cache}\n"
            f"Cohorts analysis        : {self._cohorts_analysis}\n"
            f"Site filters analysis   : {self._site_filters_analysis}\n"
            f"Software version        : malariagen_data {malariagen_data.__version__}\n"
            f"Client location         : {self._client_location}\n"
            f"---\n"
            f"Please note that data are subject to terms of use,\n"
            f"for more information see https://www.malariagen.net/data\n"
            f"or contact data@malariagen.net."
        )
        # TODO: API doc https://malariagen.github.io/vector-data/af1/api.html
        return text

    def _repr_html_(self):
        # TODO: See also the <a href="https://malariagen.github.io/vector-data/af1/api.html">Af1 API docs</a>.
        html = f"""
            <table class="malariagen-af1">
                <thead>
                    <tr>
                        <th style="text-align: left" colspan="2">MalariaGEN Af1 API client</th>
                    </tr>
                    <tr><td colspan="2" style="text-align: left">
                        Please note that data are subject to terms of use,
                        for more information see <a href="https://www.malariagen.net/data">
                        the MalariaGEN website</a> or contact data@malariagen.net.
                    </td></tr>
                </thead>
                <tbody>
                    <tr>
                        <th style="text-align: left">
                            Storage URL
                        </th>
                        <td>{self._url}</td>
                    </tr>
                    <tr>
                        <th style="text-align: left">
                            Data releases available
                        </th>
                        <td>{', '.join(self.releases)}</td>
                    </tr>
                    <tr>
                        <th style="text-align: left">
                            Results cache
                        </th>
                        <td>{self._results_cache}</td>
                    </tr>
                    <tr>
                        <th style="text-align: left">
                            Cohorts analysis
                        </th>
                        <td>{self._cohorts_analysis}</td>
                    </tr>
                    <tr>
                        <th style="text-align: left">
                            Site filters analysis
                        </th>
                        <td>{self._site_filters_analysis}</td>
                    </tr>
                    <tr>
                        <th style="text-align: left">
                            Software version
                        </th>
                        <td>malariagen_data {malariagen_data.__version__}</td>
                    </tr>
                    <tr>
                        <th style="text-align: left">
                            Client location
                        </th>
                        <td>{self._client_location}</td>
                    </tr>
                </tbody>
            </table>
        """
        return html

    # TODO: generalise (species, cohorts) so we can abstract to parent class
    def _sample_metadata(self, *, sample_set):
        df = self._read_general_metadata(sample_set=sample_set)
        df_cohorts = self._read_cohort_metadata(sample_set=sample_set)
        df = df.merge(df_cohorts, on="sample_id", sort=False)
        return df

    def _transcript_to_gene_name(self, transcript):
        df_genome_features = self.genome_features().set_index("ID")
        rec_transcript = df_genome_features.loc[transcript]
        parent = rec_transcript["Parent"]

        # E.g. manual overrides (used in Ag3)
        # if parent == "AGAP004707":
        #     parent_name = "Vgsc/para"
        # else:
        #     parent_name = rec_parent["Name"]

        # Note: Af1 doesn't have the "Name" attribute
        # rec_parent = df_genome_features.loc[parent]
        # parent_name = rec_parent["Name"]
        parent_name = parent

        return parent_name

    def _site_mask_ids(self):
        if self._site_filters_analysis == "dt_20200416":
            return ["funestus"]
        elif self._site_filters_analysis == "sc_20220908":
            return ["funestus"]
        else:
            raise ValueError

    def results_cache_get(self, *, name, params):
        debug = self._log.debug
        if self._results_cache is None:
            raise CacheMiss
        params = params.copy()
        # TODO: when cohorts are available
        # params["cohorts_analysis"] = self._cohorts_analysis
        params["site_filters_analysis"] = self._site_filters_analysis
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
        # TODO: when cohorts are available
        # params["cohorts_analysis"] = self._cohorts_analysis
        params["site_filters_analysis"] = self._site_filters_analysis
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
        name = SNP_ALLELE_COUNTS_CACHE_NAME

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

    def genome_features(
        self, region=None, attributes=("ID", "Parent", "Note", "description")
    ):
        """Access genome feature annotations.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Contig name (e.g., "2RL"), gene name (e.g., "LOC125767311"), genomic
            region defined with coordinates (e.g., "2RL:44,989,425-44,998,059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["2RL", "3RL"].
        attributes : list of str, optional
            Attribute keys to unpack into columns. Provide "*" to unpack all
            attributes.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of genome annotations, one row per feature.

        """

        # Here we override the superclass implementation in order to provide a
        # different default value for the `attributes` parameter, because the
        # genome annotations don't include a "Name" attribute but do include a
        # "Note" attribute which is probably useful to include instead.
        #
        # Also, we take the opportunity to customise the docstring to use
        # examples specific to funestus.
        #
        # See also https://github.com/malariagen/malariagen-data-python/issues/306

        return super().genome_features(region=region, attributes=attributes)

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
            Gene transcript ID, e.g., "LOC125767311_t2".
        area_by : str
            Column name in the sample metadata to use to group samples
            spatially. E.g., use "admin1_iso" or "admin1_name" to group by level
            1 administrative divisions, or use "admin2_name" to group by level 2
            administrative divisions.
        period_by : {"year", "quarter", "month"}
            Length of time to group samples temporally.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "1229-VO-GH-DADZIE-VMF00095") or a list of
            sample set identifiers (e.g., ["1240-VO-CD-KOEKEMOER-VMF00099", "1240-VO-MZ-KOEKEMOER-VMF00101"]) or a
            release identifier (e.g., "1.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'funestus' and country == 'Ghana'".
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
        variant_pass_funestus = np.repeat(
            ds_snps["variant_filter_pass_funestus"].values, 3
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
        df_variants = pd.DataFrame(
            {
                "contig": variant_contig,
                "position": variant_position,
                "ref_allele": variant_ref_allele.astype("U1"),
                "alt_allele": variant_alt_allele.astype("U1"),
                "max_af": max_af,
                "pass_funestus": variant_pass_funestus,
            }
        )

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
