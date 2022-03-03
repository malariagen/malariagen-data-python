import warnings
from bisect import bisect_left, bisect_right
from collections import Counter

import allel
import dask.array as da
import ipyleaflet
import ipywidgets
import numba
import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr
import zarr
from statsmodels.stats.proportion import proportion_confint

from . import veff
from .util import (
    DIM_ALLELE,
    DIM_PLOIDY,
    DIM_SAMPLE,
    DIM_VARIANT,
    Region,
    da_compress,
    da_from_zarr,
    dask_compress_dataset,
    init_filesystem,
    init_zarr_store,
    locate_region,
    read_gff3,
    resolve_region,
    unpack_gff3_attributes,
    xarray_concat,
)

PUBLIC_RELEASES = ("3.0",)
DEFAULT_URL = "gs://vo_agam_release/"
GENESET_GFF3_PATH = (
    "reference/genome/agamp4/Anopheles-gambiae-PEST_BASEFEATURES_AgamP4.12.gff3.gz"
)
GENOME_ZARR_PATH = (
    "reference/genome/agamp4/Anopheles-gambiae-PEST_CHROMOSOMES_AgamP4.zarr"
)
DEFAULT_SPECIES_ANALYSIS = "aim_20200422"
DEFAULT_SITE_FILTERS_ANALYSIS = "dt_20200416"
DEFAULT_COHORTS_ANALYSIS = "20211101"
CONTIGS = "2R", "2L", "3R", "3L", "X"

AA_CHANGE_QUERY = (
    "effect in ['NON_SYNONYMOUS_CODING', 'START_LOST', 'STOP_LOST', 'STOP_GAINED']"
)

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


def _release_to_path(release):
    """Compatibility function, allows us to use release identifiers like "3.0"
    and "3.1" in the public API, and map these internally into storage path
    segments."""
    if release == "3.0":
        # special case
        return "v3"
    elif release.startswith("3."):
        return f"v{release}"
    else:
        raise ValueError(f"Invalid release: {release!r}")


def _path_to_release(path):
    """Compatibility function, allows us to use release identifiers like "3.0"
    and "3.1" in the public API, and map these internally into storage path
    segments."""
    if path == "v3":
        return "3.0"
    elif path.startswith("v3."):
        return path[1:]
    else:
        raise RuntimeError(f"Unexpected release path: {path!r}")


class Ag3:
    """Provides access to data from Ag 3 releases.

    Parameters
    ----------
    url : str
        Base path to data. Give "gs://vo_agam_release/" to use Google Cloud
        Storage, or a local path on your file system if data have been
        downloaded.
    **kwargs
        Passed through to fsspec when setting up file system access.

    Examples
    --------
    Access data from Google Cloud Storage (default):

        >>> import malariagen_data
        >>> ag3 = malariagen_data.Ag3()

    Access data downloaded to a local file system:

        >>> ag3 = malariagen_data.Ag3("/local/path/to/vo_agam_release/")

    """

    contigs = CONTIGS

    def __init__(self, url=DEFAULT_URL, **kwargs):

        self._url = url
        self._pre = kwargs.pop("pre", False)

        # setup filesystem
        self._fs, self._base_path = init_filesystem(url, **kwargs)

        # setup caches
        self._cache_releases = None
        self._cache_sample_sets = dict()
        self._cache_general_metadata = dict()
        self._cache_species_calls = dict()
        self._cache_site_filters = dict()
        self._cache_snp_sites = None
        self._cache_snp_genotypes = dict()
        self._cache_genome = None
        self._cache_annotator = None
        self._cache_geneset = dict()
        self._cache_cross_metadata = None
        self._cache_site_annotations = None
        self._cache_cnv_hmm = dict()
        self._cache_cnv_coverage_calls = dict()
        self._cache_cnv_discordant_read_calls = dict()
        self._cache_haplotypes = dict()
        self._cache_haplotype_sites = dict()
        self._cache_cohort_metadata = dict()

    def __repr__(self):
        return (
            f"<MalariaGEN Ag3 data resource API>\n"
            f"Storage URL           : {self._url}\n"
            f"Releases available    : {', '.join(self.releases)}\n"
            f"Cohorts analysis      : {DEFAULT_COHORTS_ANALYSIS}\n"
            f"Species analysis      : {DEFAULT_SPECIES_ANALYSIS}\n"
            f"Site filters analysis : {DEFAULT_SITE_FILTERS_ANALYSIS}\n"
            f"---\n"
            f"Please note that data are subject to terms of use,\n"
            f"for more information see https://www.malariagen.net/data\n"
            f"or contact data@malariagen.net. For API documentation see: "
            f"https://malariagen.github.io/vector-data/ag3/api.html"
        )

    def _repr_html_(self):
        return f"""
            <table class="malariagen-ag3">
                <thead>
                    <tr>
                        <th style="text-align: left" colspan="2">MalariaGEN Ag3 data resource API</th>
                    </tr>
                    <tr><td colspan="2" style="text-align: left">
                        Please note that data are subject to terms of use,
                        for more information see <a href="https://www.malariagen.net/data">
                        the MalariaGEN website</a> or contact data@malariagen.net.
                        See also the <a href="https://malariagen.github.io/vector-data/ag3/api.html">Ag3 API docs</a>.
                    </td></tr>
                </thead>
                <tbody>
                    <tr><th style="text-align: left">Storage URL</th><td>{self._url}</td></tr>
                    <tr><th style="text-align: left">Releases available</th><td>{', '.join(self.releases)}</td></tr>
                    <tr><th style="text-align: left">Cohorts analysis</th><td>{DEFAULT_COHORTS_ANALYSIS}</td></tr>
                    <tr><th style="text-align: left">Species analysis</th><td>{DEFAULT_SPECIES_ANALYSIS}</td></tr>
                    <tr><th style="text-align: left">Site filters analysis</th><td>{DEFAULT_SITE_FILTERS_ANALYSIS}</td></tr>
                </tbody>
            </table>
        """

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
                            _path_to_release(d)
                            for d in sub_dirs
                            if d.startswith("v3")
                            and self._fs.exists(f"{self._base_path}/{d}/manifest.tsv")
                        ]
                    )
                )
                if len(releases) == 0:
                    raise ValueError("No releases found.")
                self._cache_releases = releases
            else:
                self._cache_releases = PUBLIC_RELEASES
        return self._cache_releases

    def _read_sample_sets(self, *, release):
        """Read the manifest of sample sets for a given release."""
        release_path = _release_to_path(release)
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
            Release identifier. Give "3.0" to access the Ag1000G phase 3 data
            release.

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
                return self._cache_sample_sets[release]

            except KeyError:
                df = self._read_sample_sets(release=release)
                self._cache_sample_sets[release] = df
                return df

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
            return df

        else:
            raise TypeError

    @property
    def v3_wild(self):
        # legacy, convenience property to access sample sets from the
        # 3.0 release, excluding the lab crosses
        return [
            x
            for x in self.sample_sets(release="3.0")["sample_set"].tolist()
            if x != "AG1000G-X"
        ]

    def _lookup_release(self, *, sample_set):
        """Find which release a sample set was included in."""
        df_sample_sets = self.sample_sets().set_index("sample_set")
        try:
            return df_sample_sets.loc[sample_set]["release"]
        except KeyError:
            raise ValueError(f"No release found for sample set {sample_set!r}")

    def _read_general_metadata(self, *, sample_set):
        """Read metadata for a single sample set."""
        try:
            return self._cache_general_metadata[sample_set]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            release_path = _release_to_path(release)
            path = f"{self._base_path}/{release_path}/metadata/general/{sample_set}/samples.meta.csv"
            with self._fs.open(path) as f:
                df = pd.read_csv(f, na_values="")

            # ensure all column names are lower case
            df.columns = [c.lower() for c in df.columns]

            # add a couple of columns for convenience
            df["sample_set"] = sample_set
            df["release"] = release

            self._cache_general_metadata[sample_set] = df
            return df

    def _read_species_calls(self, *, sample_set, analysis):
        """Read species calls for a single sample set."""
        key = (sample_set, analysis)
        try:
            return self._cache_species_calls[key]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            release_path = _release_to_path(release)
            path_prefix = f"{self._base_path}/{release_path}/metadata"
            if analysis == "aim_20200422":
                path = f"{path_prefix}/species_calls_20200422/{sample_set}/samples.species_aim.csv"
            elif analysis == "pca_20200422":
                path = f"{path_prefix}/species_calls_20200422/{sample_set}/samples.species_pca.csv"
            else:
                raise ValueError(f"Unknown species calling analysis: {analysis!r}")
            with self._fs.open(path) as f:
                df = pd.read_csv(
                    f,
                    na_values="",
                    # ensure correct dtype even where all values are missing
                    dtype={
                        "species_gambcolu_arabiensis": object,
                        "species_gambiae_coluzzii": object,
                    },
                )

            # add a single species call column, for convenience
            def consolidate_species(s):
                species_gambcolu_arabiensis = s["species_gambcolu_arabiensis"]
                species_gambiae_coluzzii = s["species_gambiae_coluzzii"]
                if species_gambcolu_arabiensis == "arabiensis":
                    return "arabiensis"
                elif species_gambcolu_arabiensis == "intermediate":
                    return "intermediate_arabiensis_gambiae"
                elif species_gambcolu_arabiensis == "gamb_colu":
                    # look at gambiae_vs_coluzzii
                    if species_gambiae_coluzzii == "gambiae":
                        return "gambiae"
                    elif species_gambiae_coluzzii == "coluzzii":
                        return "coluzzii"
                    elif species_gambiae_coluzzii == "intermediate":
                        return "intermediate_gambiae_coluzzii"
                else:
                    # some individuals, e.g., crosses, have a missing species call
                    return np.nan

            df["species"] = df.apply(consolidate_species, axis=1)

            if analysis == "aim_20200422":
                # normalise column prefixes
                df = df.rename(
                    columns={
                        "aim_fraction_arab": "aim_species_fraction_arab",
                        "aim_fraction_colu": "aim_species_fraction_colu",
                        "species_gambcolu_arabiensis": "aim_species_gambcolu_arabiensis",
                        "species_gambiae_coluzzii": "aim_species_gambiae_coluzzii",
                        "species": "aim_species",
                    }
                )
            elif analysis == "pca_20200422":
                # normalise column prefixes
                df = df.rename(
                    # normalise column prefixes
                    columns={
                        "PC1": "pca_species_PC1",
                        "PC2": "pca_species_PC2",
                        "species_gambcolu_arabiensis": "pca_species_gambcolu_arabiensis",
                        "species_gambiae_coluzzii": "pca_species_gambiae_coluzzii",
                        "species": "pca_species",
                    }
                )

            # ensure all column names are lower case
            df.columns = [c.lower() for c in df.columns]

            self._cache_species_calls[key] = df
            return df

    def _prep_sample_sets_arg(self, *, sample_sets):
        """Common handling for the `sample_sets` parameter. For convenience, we
        allow this to be a single sample set, or a list of sample sets, or a
        release identifier, or a list of release identifiers."""

        if sample_sets is None:
            # all available sample sets
            sample_sets = self.sample_sets()["sample_set"].tolist()

        elif isinstance(sample_sets, str):

            if sample_sets.startswith("3."):
                # convenience, can use a release identifier to denote all sample sets
                # in a release
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

    def species_calls(self, sample_sets=None, analysis=DEFAULT_SPECIES_ANALYSIS):
        """Access species calls for one or more sample sets.

        Parameters
        ----------
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"] or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        analysis : {"aim_20200422", "pca_20200422"}
            Species calling analysis.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of species calls for one or more sample sets, one row
            per sample.

        """

        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)

        # concatenate multiple sample sets
        dfs = [
            self._read_species_calls(sample_set=s, analysis=analysis)
            for s in sample_sets
        ]
        df = pd.concat(dfs, axis=0, ignore_index=True)

        return df

    def _sample_metadata(self, *, sample_set, species_analysis, cohorts_analysis):
        df = self._read_general_metadata(sample_set=sample_set)
        if species_analysis is not None:
            df_species = self._read_species_calls(
                sample_set=sample_set, analysis=species_analysis
            )
            df = df.merge(df_species, on="sample_id", sort=False)
        if cohorts_analysis is not None:
            df_cohorts = self.sample_cohorts(
                sample_sets=sample_set, cohorts_analysis=cohorts_analysis
            )
            df = df.merge(df_cohorts, on="sample_id", sort=False)
        return df

    def sample_metadata(
        self,
        sample_sets=None,
        species_analysis=DEFAULT_SPECIES_ANALYSIS,
        cohorts_analysis=DEFAULT_COHORTS_ANALYSIS,
    ):
        """Access sample metadata for one or more sample sets.

        Parameters
        ----------
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        species_analysis : {"aim_20200422", "pca_20200422"}, optional
            Include species calls in metadata.
        cohorts_analysis : str, optional
            Cohort analysis identifier (date of analysis), optional,  default is
            the latest version. Includes sample cohort calls in metadata.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of sample metadata, one row per sample.

        """

        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)

        # concatenate multiple sample sets
        dfs = [
            self._sample_metadata(
                sample_set=s,
                species_analysis=species_analysis,
                cohorts_analysis=cohorts_analysis,
            )
            for s in sample_sets
        ]
        df = pd.concat(dfs, axis=0, ignore_index=True)

        return df

    def open_site_filters(self, mask, analysis=DEFAULT_SITE_FILTERS_ANALYSIS):
        """Open site filters zarr.

        Parameters
        ----------
        mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Mask to use.
        analysis : str, optional
            Site filters analysis version.

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        key = mask, analysis
        try:
            return self._cache_site_filters[key]
        except KeyError:
            path = f"{self._base_path}/v3/site_filters/{analysis}/{mask}/"
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_site_filters[key] = root
            return root

    def _site_filters(
        self,
        *,
        region,
        mask,
        field,
        analysis,
        inline_array,
        chunks,
    ):
        assert isinstance(region, Region)
        root = self.open_site_filters(mask=mask, analysis=analysis)
        z = root[f"{region.contig}/variants/{field}"]
        d = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        if region.start or region.end:
            pos = self.snp_sites(region=region.contig, field="POS")
            loc_region = locate_region(region, pos)
            d = d[loc_region]
        return d

    def site_filters(
        self,
        region,
        mask,
        field="filter_pass",
        analysis=DEFAULT_SITE_FILTERS_ANALYSIS,
        inline_array=True,
        chunks="native",
    ):
        """Access SNP site filters.

        Parameters
        ----------
        region: str or list of str or Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "AGAP005958"].
        mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Mask to use.
        field : str, optional
            Array to access.
        analysis : str, optional
            Site filters analysis version.
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

        region = resolve_region(self, region)
        if isinstance(region, Region):
            region = [region]

        d = da.concatenate(
            [
                self._site_filters(
                    region=r,
                    mask=mask,
                    field=field,
                    analysis=analysis,
                    inline_array=inline_array,
                    chunks=chunks,
                )
                for r in region
            ]
        )

        return d

    def open_snp_sites(self):
        """Open SNP sites zarr.

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        if self._cache_snp_sites is None:
            path = f"{self._base_path}/v3/snp_genotypes/all/sites/"
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_snp_sites = root
        return self._cache_snp_sites

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
            pos = root[f"{region.contig}/variants/POS"]
            loc_region = locate_region(region, pos)
            ret = ret[loc_region]
        return ret

    def snp_sites(
        self,
        region,
        field,
        site_mask=None,
        site_filters_analysis=DEFAULT_SITE_FILTERS_ANALYSIS,
        inline_array=True,
        chunks="native",
    ):
        """Access SNP site data (positions and alleles).

        Parameters
        ----------
        region: str or list of str or Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "AGAP005958"].
        field : {"POS", "REF", "ALT"}
            Array to access.
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_filters_analysis : str
            Site filters analysis version.
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

        region = resolve_region(self, region)
        if isinstance(region, Region):
            region = [region]

        # concatenate
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

        if site_mask is not None:
            loc_sites = self.site_filters(
                region=region,
                mask=site_mask,
                analysis=site_filters_analysis,
                chunks=chunks,
                inline_array=inline_array,
            )
            ret = da_compress(loc_sites, ret, axis=0)

        return ret

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
            release_path = _release_to_path(release)
            path = f"{self._base_path}/{release_path}/snp_genotypes/all/{sample_set}/"
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_snp_genotypes[sample_set] = root
            return root

    def _snp_genotypes(self, *, region, sample_set, field, inline_array, chunks):
        # single contig, single sample set
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
        field="GT",
        site_mask=None,
        site_filters_analysis=DEFAULT_SITE_FILTERS_ANALYSIS,
        inline_array=True,
        chunks="native",
    ):
        """Access SNP genotypes and associated data.

        Parameters
        ----------
        region: str or list of str or Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "AGAP005958"].
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        field : {"GT", "GQ", "AD", "MQ"}
            Array to access.
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_filters_analysis : str, optional
            Site filters analysis version.
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

        # normalise parameters
        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)
        region = resolve_region(self, region)

        # normalise region to list to simplify concatenation logic
        if isinstance(region, Region):
            region = [region]

        # concatenate multiple sample sets and/or contigs
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

            # concatenate data from multiple sample sets
            x = da.concatenate(ly, axis=1)

            # locate region - do this only once, optimisation
            if r.start or r.end:
                pos = self.snp_sites(region=r.contig, field="POS")
                loc_region = locate_region(r, pos)
                x = x[loc_region]

            lx.append(x)

        # concatenate data from multiple regions
        d = da.concatenate(lx, axis=0)

        # apply site filters if requested
        if site_mask is not None:
            loc_sites = self.site_filters(
                region=region, mask=site_mask, analysis=site_filters_analysis
            )
            d = da_compress(loc_sites, d, axis=0)

        return d

    def open_genome(self):
        """Open the reference genome zarr.

        Returns
        -------
        root : zarr.hierarchy.Group
            Zarr hierarchy containing the reference genome sequence.

        """
        if self._cache_genome is None:
            path = f"{self._base_path}/{GENOME_ZARR_PATH}"
            store = init_zarr_store(fs=self._fs, path=path)
            self._cache_genome = zarr.open_consolidated(store=store)
        return self._cache_genome

    def genome_sequence(self, region, inline_array=True, chunks="native"):
        """Access the reference genome sequence.

        Parameters
        ----------
        region: str or list of str or Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
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
        region = resolve_region(self, region)
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

    def geneset(self, attributes=("ID", "Parent", "Name", "description")):
        """Access genome feature annotations (AgamP4.12).

        Parameters
        ----------
        attributes : list of str, optional
            Attribute keys to unpack into columns. Provide "*" to unpack all
            attributes.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of genome annotations, one row per feature.

        """

        if attributes is not None:
            attributes = tuple(attributes)

        try:
            df = self._cache_geneset[attributes]

        except KeyError:
            path = f"{self._base_path}/{GENESET_GFF3_PATH}"
            with self._fs.open(path, mode="rb") as f:
                df = read_gff3(f, compression="gzip")
            if attributes is not None:
                df = unpack_gff3_attributes(df, attributes=attributes)
            self._cache_geneset[attributes] = df

        return df

    def _transcript_to_gene_name(self, transcript):
        df_geneset = self.geneset().set_index("ID")
        rec_transcript = df_geneset.loc[transcript]
        parent = rec_transcript["Parent"]
        rec_parent = df_geneset.loc[parent]

        # manual overrides
        if parent == "AGAP004707":
            parent_name = "Vgsc/para"
        else:
            parent_name = rec_parent["Name"]

        return parent_name

    def is_accessible(
        self, region, site_mask, site_filters_analysis=DEFAULT_SITE_FILTERS_ANALYSIS
    ):
        """Compute genome accessibility array.

        Parameters
        ----------
        region: str or list of str or Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_filters_analysis : str, optional
            Site filters analysis version.

        Returns
        -------
        a : numpy.ndarray
            An array of boolean values identifying accessible genome sites.

        """

        # resolve region
        region = resolve_region(self, region)

        # determine contig sequence length
        seq_length = self.genome_sequence(region).shape[0]

        # setup output
        is_accessible = np.zeros(seq_length, dtype=bool)

        pos = self.snp_sites(region, field="POS").compute()
        if region.start:
            offset = region.start
        else:
            offset = 1

        # access site filters
        filter_pass = self.site_filters(
            region=region, mask=site_mask, analysis=site_filters_analysis
        ).compute()

        # assign values from site filters
        is_accessible[pos - offset] = filter_pass

        return is_accessible

    @staticmethod
    def _site_mask_ids(*, site_filters_analysis):
        if site_filters_analysis == "dt_20200416":
            return "gamb_colu_arab", "gamb_colu", "arab"
        else:
            raise ValueError

    def _snp_df(self, *, transcript, site_filters_analysis):
        """Set up a dataframe with SNP site and filter columns."""

        # get feature direct from geneset
        gs = self.geneset()
        feature = gs[gs["ID"] == transcript].squeeze()
        contig = feature.contig
        region = Region(contig, feature.start, feature.end)

        # grab pos, ref and alt for chrom arm from snp_sites
        pos = self.snp_sites(region=contig, field="POS")
        ref = self.snp_sites(region=contig, field="REF")
        alt = self.snp_sites(region=contig, field="ALT")
        loc_feature = locate_region(region, pos)
        pos = pos[loc_feature].compute()
        ref = ref[loc_feature].compute()
        alt = alt[loc_feature].compute()

        # access site filters
        filter_pass = dict()
        masks = self._site_mask_ids(site_filters_analysis=site_filters_analysis)
        for m in masks:
            x = self.site_filters(region=contig, mask=m, analysis=site_filters_analysis)
            x = x[loc_feature].compute()
            filter_pass[m] = x

        # setup columns with contig, pos, ref, alt columns
        cols = {
            "contig": contig,
            "position": np.repeat(pos, 3),
            "ref_allele": np.repeat(ref.astype("U1"), 3),
            "alt_allele": alt.astype("U1").flatten(),
        }

        # add mask columns
        for m in masks:
            x = filter_pass[m]
            cols[f"pass_{m}"] = np.repeat(x, 3)

        # construct dataframe
        df_snps = pd.DataFrame(cols)

        return region, df_snps

    def _annotator(self):
        # setup variant effect annotator
        if self._cache_annotator is None:
            self._cache_annotator = veff.Annotator(
                genome=self.open_genome(), geneset=self.geneset()
            )
        return self._cache_annotator

    def snp_effects(
        self,
        transcript,
        site_mask=None,
        site_filters_analysis=DEFAULT_SITE_FILTERS_ANALYSIS,
    ):
        """Compute variant effects for a gene transcript.

        Parameters
        ----------
        transcript : str
            Gene transcript ID (AgamP4.12), e.g., "AGAP004707-RA".
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}, optional
            Site filters mask to apply.
        site_filters_analysis : str, optional
            Site filters analysis version.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of all possible SNP variants and their effects, one row
            per variant.

        """

        # setup initial dataframe of SNPs
        _, df_snps = self._snp_df(
            transcript=transcript, site_filters_analysis=site_filters_analysis
        )

        # setup variant effect annotator
        ann = self._annotator()

        # apply mask if requested
        if site_mask is not None:
            loc_sites = df_snps[f"pass_{site_mask}"]
            df_snps = df_snps.loc[loc_sites]

        # reset index after filtering
        df_snps.reset_index(inplace=True, drop=True)

        # add effects to the dataframe
        ann.get_effects(transcript=transcript, variants=df_snps)

        return df_snps

    def _prep_cohorts_arg(
        self, *, cohorts, sample_sets, species_analysis, cohorts_analysis
    ):

        # build cohort dictionary where key=cohort_id, value=loc_coh
        coh_dict = {}
        if isinstance(cohorts, dict):
            # get sample metadata
            df_meta = self.sample_metadata(
                sample_sets=sample_sets, species_analysis=species_analysis
            )
            for coh, query in cohorts.items():
                # locate samples
                loc_coh = df_meta.eval(query).values
                coh_dict[coh] = loc_coh
        if isinstance(cohorts, str):
            # grab the cohorts dataframe
            df_coh = self.sample_cohorts(
                sample_sets=sample_sets, cohorts_analysis=cohorts_analysis
            )
            # fix the string to match columns
            cohorts = "cohort_" + cohorts
            # check the given cohort set exists
            if cohorts not in df_coh.columns:
                raise ValueError(f"{cohorts!r} is not a known cohort set")
            cohort_labels = df_coh[cohorts].unique()
            # remove the nans and sort
            cohort_labels = sorted([c for c in cohort_labels if isinstance(c, str)])
            for coh in cohort_labels:
                loc_coh = df_coh[cohorts] == coh
                coh_dict[coh] = loc_coh.values

        return coh_dict

    def snp_allele_frequencies(
        self,
        transcript,
        cohorts,
        sample_query=None,
        cohorts_analysis=DEFAULT_COHORTS_ANALYSIS,
        min_cohort_size=10,
        site_mask=None,
        site_filters_analysis=DEFAULT_SITE_FILTERS_ANALYSIS,
        species_analysis=DEFAULT_SPECIES_ANALYSIS,
        sample_sets=None,
        drop_invariant=True,
        effects=True,
    ):
        """Compute per variant allele frequencies for a gene transcript.

        Parameters
        ----------
        transcript : str
            Gene transcript ID (AgamP4.12), e.g., "AGAP004707-RD".
        cohorts : str or dict
            If a string, gives the name of a predefined cohort set, e.g., one of
            {"admin1_month", "admin1_year", "admin2_month", "admin2_year"}.
            If a dict, should map cohort labels to sample queries, e.g.,
            `{"bf_2012_col": "country == 'Burkina Faso' and year == 2012 and taxon == 'coluzzii'"}`.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample metadata e.g.,
            "taxon == 'coluzzii' and country == 'Burkina Faso'".
        cohorts_analysis : str
            Cohort analysis version, default is the latest version.
        min_cohort_size : int
            Minimum cohort size. Any cohorts below this size are omitted.
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_filters_analysis : str, optional
            Site filters analysis version.
        species_analysis : {"aim_20200422", "pca_20200422"}, optional
            Species calls analysis version.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "3.0") or a list of release identifiers.
        drop_invariant : bool, optional
            If True, variants with no alternate allele calls in any cohorts are dropped from
            the result.
        effects : bool, optional
            If True, add SNP effect columns.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of SNP frequencies, one row per variant.

        Notes
        -----
        Cohorts with fewer samples than min_cohort_size will be excluded from output.

        """

        # check parameters
        _check_param_min_cohort_size(min_cohort_size)

        # access sample metadata
        df_samples = self.sample_metadata(
            sample_sets=sample_sets,
            cohorts_analysis=cohorts_analysis,
            species_analysis=species_analysis,
        )

        # handle sample_query
        loc_samples = None
        if sample_query is not None:
            loc_samples = df_samples.eval(sample_query).values

        # setup initial dataframe of SNPs
        region, df_snps = self._snp_df(
            transcript=transcript, site_filters_analysis=site_filters_analysis
        )

        # get genotypes
        gt = self.snp_genotypes(
            region=region,
            sample_sets=sample_sets,
            field="GT",
        )

        # slice to feature location
        gt = gt.compute()

        # build coh dict
        coh_dict = self._prep_cohorts_arg(
            cohorts=cohorts,
            sample_sets=sample_sets,
            species_analysis=species_analysis,
            cohorts_analysis=cohorts_analysis,
        )

        # count alleles
        freq_cols = dict()
        for coh, loc_coh in coh_dict.items():
            # handle sample query
            if loc_samples is not None:
                loc_coh = loc_coh & loc_samples
            n_samples = np.count_nonzero(loc_coh)
            if n_samples >= min_cohort_size:
                gt_coh = np.compress(loc_coh, gt, axis=1)
                # count alleles
                ac_coh = allel.GenotypeArray(gt_coh).count_alleles(max_allele=3)
                # compute allele frequencies
                af_coh = ac_coh.to_frequencies()
                # add column to dict
                freq_cols["frq_" + coh] = af_coh[:, 1:].flatten()

        # build a dataframe with the frequency columns
        df_freqs = pd.DataFrame(freq_cols)

        # compute max_af
        df_max_af = pd.DataFrame({"max_af": df_freqs.max(axis=1)})

        # build the final dataframe
        df_snps.reset_index(drop=True, inplace=True)
        df_snps = pd.concat([df_snps, df_freqs, df_max_af], axis=1)

        # apply site mask if requested
        if site_mask is not None:
            loc_sites = df_snps[f"pass_{site_mask}"]
            df_snps = df_snps.loc[loc_sites]

        # drop invariants
        if drop_invariant:
            loc_variant = df_snps["max_af"] > 0
            df_snps = df_snps.loc[loc_variant]

        # reset index after filtering
        df_snps.reset_index(inplace=True, drop=True)

        # add effects
        if effects:

            # add effect annotations
            ann = self._annotator()
            ann.get_effects(transcript=transcript, variants=df_snps)

            # add label
            df_snps["label"] = df_snps.apply(_make_snp_label_effect, axis="columns")

            # set index
            df_snps.set_index(
                ["contig", "position", "ref_allele", "alt_allele", "aa_change"],
                inplace=True,
            )

        else:

            # add label
            df_snps["label"] = df_snps.apply(_make_snp_label, axis="columns")

            # set index
            df_snps.set_index(
                ["contig", "position", "ref_allele", "alt_allele"],
                inplace=True,
            )

        # add metadata
        gene_name = self._transcript_to_gene_name(transcript)
        title = transcript
        if gene_name:
            title += f" ({gene_name})"
        title += " SNP frequencies"
        df_snps.attrs["title"] = title

        return df_snps

    def cross_metadata(self):
        """Load a dataframe containing metadata about samples in colony crosses, including
        which samples are parents or progeny in which crosses.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of sample metadata for colony crosses.

        """

        if self._cache_cross_metadata is None:

            path = f"{self._base_path}/v3/metadata/crosses/crosses.fam"
            fam_names = [
                "cross",
                "sample_id",
                "father_id",
                "mother_id",
                "sex",
                "phenotype",
            ]
            with self._fs.open(path) as f:
                df = pd.read_csv(
                    f,
                    sep="\t",
                    na_values=["", "0"],
                    names=fam_names,
                    dtype={"sex": str},
                )

            # convert "sex" column for consistency with sample metadata
            df.loc[df["sex"] == "1", "sex"] = "M"
            df.loc[df["sex"] == "2", "sex"] = "F"

            # add a "role" column for convenience
            df["role"] = "progeny"
            df.loc[df["mother_id"].isna(), "role"] = "parent"

            # drop "phenotype" column, not used
            df.drop("phenotype", axis="columns", inplace=True)

            self._cache_cross_metadata = df

        return self._cache_cross_metadata

    def open_site_annotations(self):
        """Open site annotations zarr.

        Returns
        -------
        root : zarr.hierarchy.Group

        """

        if self._cache_site_annotations is None:
            path = f"{self._base_path}/reference/genome/agamp4/Anopheles-gambiae-PEST_SEQANNOTATION_AgamP4.12.zarr"
            store = init_zarr_store(fs=self._fs, path=path)
            self._cache_site_annotations = zarr.open_consolidated(store=store)
        return self._cache_site_annotations

    def site_annotations(
        self,
        region,
        field,
        site_mask=None,
        site_filters_analysis=DEFAULT_SITE_FILTERS_ANALYSIS,
        inline_array=True,
        chunks="native",
    ):
        """Load site annotations.

        Parameters
        ----------
        region: str or list of str or Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic region
            defined with coordinates (e.g., "2L:44989425-44998059") or a named tuple with
            genomic location `Region(contig, start, end)`.
        field : str
            One of "codon_degeneracy", "codon_nonsyn", "codon_position", "seq_cls",
            "seq_flen", "seq_relpos_start", "seq_relpos_stop".
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_filters_analysis : str
            Site filters analysis version.
        inline_array : bool, optional
            Passed through to dask.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr chunks.
            Also can be a target size, e.g., '200 MiB'.

        Returns
        -------
        d : dask.Array
            An array of site annotations.

        """

        # access the array of values for all genome positions
        root = self.open_site_annotations()

        # resolve region
        region = resolve_region(self, region)
        if isinstance(region, list):
            raise TypeError("Multiple regions not supported.")

        d = da_from_zarr(
            root[field][region.contig], inline_array=inline_array, chunks=chunks
        )

        # access and subset to SNP positions
        pos = self.snp_sites(
            region=region,
            field="POS",
            site_mask=site_mask,
            site_filters_analysis=site_filters_analysis,
        )
        d = da.take(d, pos - 1)

        return d

    def _snp_calls_dataset(
        self, *, contig, sample_set, site_filters_analysis, inline_array, chunks
    ):

        # assert isinstance(region, Region)
        # contig = region.contig

        coords = dict()
        data_vars = dict()

        # variant arrays
        sites_root = self.open_snp_sites()

        # variant_position
        pos_z = sites_root[f"{contig}/variants/POS"]
        variant_position = da_from_zarr(pos_z, inline_array=inline_array, chunks=chunks)
        coords["variant_position"] = [DIM_VARIANT], variant_position

        # variant_allele
        ref_z = sites_root[f"{contig}/variants/REF"]
        alt_z = sites_root[f"{contig}/variants/ALT"]
        ref = da_from_zarr(ref_z, inline_array=inline_array, chunks=chunks)
        alt = da_from_zarr(alt_z, inline_array=inline_array, chunks=chunks)
        variant_allele = da.concatenate([ref[:, None], alt], axis=1)
        data_vars["variant_allele"] = [DIM_VARIANT, DIM_ALLELE], variant_allele

        # variant_contig
        contig_index = self.contigs.index(contig)
        variant_contig = da.full_like(
            variant_position, fill_value=contig_index, dtype="u1"
        )
        coords["variant_contig"] = [DIM_VARIANT], variant_contig

        # site filters arrays
        for mask in "gamb_colu_arab", "gamb_colu", "arab":
            filters_root = self.open_site_filters(
                mask=mask, analysis=site_filters_analysis
            )
            z = filters_root[f"{contig}/variants/filter_pass"]
            d = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
            data_vars[f"variant_filter_pass_{mask}"] = [DIM_VARIANT], d

        # call arrays
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

        # sample arrays
        z = calls_root["samples"]
        sample_id = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        # decode to str, as it is stored as bytes objects
        sample_id = sample_id.astype("U")
        coords["sample_id"] = [DIM_SAMPLE], sample_id

        # setup attributes
        attrs = {"contigs": self.contigs}

        # create a dataset
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        return ds

    def snp_calls(
        self,
        region,
        sample_sets=None,
        site_mask=None,
        site_filters_analysis=DEFAULT_SITE_FILTERS_ANALYSIS,
        inline_array=True,
        chunks="native",
    ):
        """Access SNP sites, site filters and genotype calls.

        Parameters
        ----------
        region: str or list of str or Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic region
            defined with coordinates (e.g., "2L:44989425-44998059") or a named tuple with
            genomic location `Region(contig, start, end)`. Multiple values can be provided
            as a list, in which case data will be concatenated, e.g., ["3R", "AGAP005958"].
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "3.0") or a list of release identifiers.
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_filters_analysis : str
            Site filters analysis version.
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr chunks.
            Also can be a target size, e.g., '200 MiB'.

        Returns
        -------
        ds : xarray.Dataset
            A dataset containing SNP sites, site filters and genotype calls.

        """

        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)
        region = resolve_region(self, region)

        # normalise to simplify concatenation logic
        if isinstance(region, Region):
            region = [region]

        # concatenate multiple sample sets and/or regions
        lx = []
        for r in region:

            ly = []
            for s in sample_sets:
                y = self._snp_calls_dataset(
                    contig=r.contig,
                    sample_set=s,
                    site_filters_analysis=site_filters_analysis,
                    inline_array=inline_array,
                    chunks=chunks,
                )
                ly.append(y)

            # concatenate data from multiple sample sets
            x = xarray_concat(ly, dim=DIM_SAMPLE)

            # handle region, do this only once - optimisation
            if r.start or r.end:
                pos = x["variant_position"].values
                loc_region = locate_region(r, pos)
                x = x.isel(variants=loc_region)

            lx.append(x)

        # concatenate data from multiple regions
        ds = xarray_concat(lx, dim=DIM_VARIANT)

        # apply site filters
        if site_mask is not None:
            ds = dask_compress_dataset(
                ds, indexer=f"variant_filter_pass_{site_mask}", dim=DIM_VARIANT
            )

        # add call_genotype_mask
        ds["call_genotype_mask"] = ds["call_genotype"] < 0

        return ds

    def snp_dataset(self, *args, **kwargs):
        # backwards compatibility, this method has been renamed to snp_calls()
        return self.snp_calls(*args, **kwargs)

    def open_cnv_hmm(self, sample_set):
        """Open CNV HMM zarr.

        Parameters
        ----------
        sample_set : str

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        try:
            return self._cache_cnv_hmm[sample_set]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            release_path = _release_to_path(release)
            path = f"{self._base_path}/{release_path}/cnv/{sample_set}/hmm/zarr"
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_cnv_hmm[sample_set] = root
        return root

    def _cnv_hmm_dataset(self, *, contig, sample_set, inline_array, chunks):

        coords = dict()
        data_vars = dict()

        # open zarr
        root = self.open_cnv_hmm(sample_set=sample_set)

        # variant arrays
        pos = root[f"{contig}/variants/POS"]
        coords["variant_position"] = (
            [DIM_VARIANT],
            da_from_zarr(pos, inline_array=inline_array, chunks=chunks),
        )
        coords["variant_end"] = (
            [DIM_VARIANT],
            da_from_zarr(
                root[f"{contig}/variants/END"], inline_array=inline_array, chunks=chunks
            ),
        )

        contig_index = self.contigs.index(contig)
        coords["variant_contig"] = (
            [DIM_VARIANT],
            da.full_like(pos, fill_value=contig_index, dtype="u1"),
        )

        # call arrays
        data_vars["call_CN"] = (
            [DIM_VARIANT, DIM_SAMPLE],
            da_from_zarr(
                root[f"{contig}/calldata/CN"], inline_array=inline_array, chunks=chunks
            ),
        )
        data_vars["call_RawCov"] = (
            [DIM_VARIANT, DIM_SAMPLE],
            da_from_zarr(
                root[f"{contig}/calldata/RawCov"],
                inline_array=inline_array,
                chunks=chunks,
            ),
        )
        data_vars["call_NormCov"] = (
            [DIM_VARIANT, DIM_SAMPLE],
            da_from_zarr(
                root[f"{contig}/calldata/NormCov"],
                inline_array=inline_array,
                chunks=chunks,
            ),
        )

        # sample arrays
        coords["sample_id"] = (
            [DIM_SAMPLE],
            da_from_zarr(root["samples"], inline_array=inline_array, chunks=chunks),
        )
        for field in "sample_coverage_variance", "sample_is_high_variance":
            data_vars[field] = (
                [DIM_SAMPLE],
                da_from_zarr(root[field], inline_array=inline_array, chunks=chunks),
            )

        # setup attributes
        attrs = {"contigs": self.contigs}

        # create a dataset
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        return ds

    def cnv_hmm(
        self,
        contig,
        sample_sets=None,
        inline_array=True,
        chunks="native",
    ):
        """Access CNV HMM data from CNV calling.

        Parameters
        ----------
        contig : str
            Chromosome arm, e.g., "3R". Multiple values can be provided
            as a list, in which case data will be concatenated, e.g., ["2R", "3R"].
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "3.0") or a list of release identifiers.
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr chunks.
            Also can be a target size, e.g., '200 MiB'.

        Returns
        -------
        ds : xarray.Dataset
            A dataset of CNV HMM calls and associated data.

        """

        # normalise parameters
        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)
        if isinstance(contig, str):
            contig = [contig]

        # concatenate
        lx = []
        for c in contig:

            ly = []
            for s in sample_sets:
                y = self._cnv_hmm_dataset(
                    contig=c,
                    sample_set=s,
                    inline_array=inline_array,
                    chunks=chunks,
                )
                ly.append(y)

            x = xarray_concat(ly, dim=DIM_SAMPLE)
            lx.append(x)

        ds = xarray_concat(lx, dim=DIM_VARIANT)

        return ds

    def open_cnv_coverage_calls(self, sample_set, analysis):
        """Open CNV coverage calls zarr.

        Parameters
        ----------
        sample_set : str
        analysis : {'gamb_colu', 'arab', 'crosses'}

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        key = (sample_set, analysis)
        try:
            return self._cache_cnv_coverage_calls[key]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            release_path = _release_to_path(release)
            path = f"{self._base_path}/{release_path}/cnv/{sample_set}/coverage_calls/{analysis}/zarr"
            # N.B., not all sample_set/analysis combinations exist, need to check
            marker = path + "/.zmetadata"
            if not self._fs.exists(marker):
                raise ValueError(
                    f"analysis f{analysis!r} not implemented for sample set {sample_set!r}"
                )
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_cnv_coverage_calls[key] = root
        return root

    def _cnv_coverage_calls_dataset(
        self,
        *,
        contig,
        sample_set,
        analysis,
        inline_array,
        chunks,
    ):

        coords = dict()
        data_vars = dict()

        # open zarr
        root = self.open_cnv_coverage_calls(sample_set=sample_set, analysis=analysis)

        # variant arrays
        pos = root[f"{contig}/variants/POS"]
        coords["variant_position"] = (
            [DIM_VARIANT],
            da_from_zarr(pos, inline_array=inline_array, chunks=chunks),
        )
        coords["variant_end"] = (
            [DIM_VARIANT],
            da_from_zarr(
                root[f"{contig}/variants/END"], inline_array=inline_array, chunks=chunks
            ),
        )
        contig_index = self.contigs.index(contig)
        coords["variant_contig"] = (
            [DIM_VARIANT],
            da.full_like(pos, fill_value=contig_index, dtype="u1"),
        )
        coords["variant_id"] = (
            [DIM_VARIANT],
            da_from_zarr(
                root[f"{contig}/variants/ID"], inline_array=inline_array, chunks=chunks
            ),
        )
        data_vars["variant_CIPOS"] = (
            [DIM_VARIANT],
            da_from_zarr(
                root[f"{contig}/variants/CIPOS"],
                inline_array=inline_array,
                chunks=chunks,
            ),
        )
        data_vars["variant_CIEND"] = (
            [DIM_VARIANT],
            da_from_zarr(
                root[f"{contig}/variants/CIEND"],
                inline_array=inline_array,
                chunks=chunks,
            ),
        )
        data_vars["variant_filter_pass"] = (
            [DIM_VARIANT],
            da_from_zarr(
                root[f"{contig}/variants/FILTER_PASS"],
                inline_array=inline_array,
                chunks=chunks,
            ),
        )

        # call arrays
        data_vars["call_genotype"] = (
            [DIM_VARIANT, DIM_SAMPLE],
            da_from_zarr(
                root[f"{contig}/calldata/GT"], inline_array=inline_array, chunks=chunks
            ),
        )

        # sample arrays
        coords["sample_id"] = (
            [DIM_SAMPLE],
            da_from_zarr(root["samples"], inline_array=inline_array, chunks=chunks),
        )

        # setup attributes
        attrs = {"contigs": self.contigs}

        # create a dataset
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        return ds

    def cnv_coverage_calls(
        self,
        contig,
        sample_set,
        analysis,
        inline_array=True,
        chunks="native",
    ):
        """Access CNV HMM data from genome-wide CNV discovery and filtering.

        Parameters
        ----------
        contig : str
            Chromosome arm, e.g., "3R". Multiple values can be provided
            as a list, in which case data will be concatenated, e.g., ["2R", "3R"].
        sample_set : str
            Sample set identifier.
        analysis : {'gamb_colu', 'arab', 'crosses'}
            Name of CNV analysis.
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr chunks.
            Also, can be a target size, e.g., '200 MiB'.

        Returns
        -------
        ds : xarray.Dataset
            A dataset of CNV alleles and genotypes.

        """

        # N.B., we cannot concatenate multiple sample sets here, because
        # different sample sets may have different sets of alleles, as the
        # calling is done independently in different sample sets.

        # normalise parameters
        if isinstance(contig, str):
            contig = [contig]

        # concatenate
        lx = []
        for c in contig:
            x = self._cnv_coverage_calls_dataset(
                contig=c,
                sample_set=sample_set,
                analysis=analysis,
                inline_array=inline_array,
                chunks=chunks,
            )
            lx.append(x)
        ds = xarray_concat(lx, dim=DIM_VARIANT)

        return ds

    def open_cnv_discordant_read_calls(self, sample_set):
        """Open CNV discordant read calls zarr.

        Parameters
        ----------
        sample_set : str

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        try:
            return self._cache_cnv_discordant_read_calls[sample_set]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            release_path = _release_to_path(release)
            path = f"{self._base_path}/{release_path}/cnv/{sample_set}/discordant_read_calls/zarr"
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_cnv_discordant_read_calls[sample_set] = root
        return root

    def _cnv_discordant_read_calls_dataset(
        self, *, contig, sample_set, inline_array, chunks
    ):

        coords = dict()
        data_vars = dict()

        # open zarr
        root = self.open_cnv_discordant_read_calls(sample_set=sample_set)

        # not all contigs have CNVs, need to check
        # TODO consider returning dataset with zero length variants dimension, would
        # probably simplify downstream logic
        if contig not in root:
            raise ValueError(f"no CNVs available for contig {contig!r}")

        # variant arrays
        pos = root[f"{contig}/variants/POS"]
        coords["variant_position"] = (
            [DIM_VARIANT],
            da_from_zarr(pos, inline_array=inline_array, chunks=chunks),
        )
        coords["variant_end"] = (
            [DIM_VARIANT],
            da_from_zarr(
                root[f"{contig}/variants/END"], inline_array=inline_array, chunks=chunks
            ),
        )
        coords["variant_id"] = (
            [DIM_VARIANT],
            da_from_zarr(
                root[f"{contig}/variants/ID"], inline_array=inline_array, chunks=chunks
            ),
        )
        contig_index = self.contigs.index(contig)
        coords["variant_contig"] = (
            [DIM_VARIANT],
            da.full_like(pos, fill_value=contig_index, dtype="u1"),
        )
        for field in "Region", "StartBreakpointMethod", "EndBreakpointMethod":
            data_vars[f"variant_{field}"] = (
                [DIM_VARIANT],
                da_from_zarr(
                    root[f"{contig}/variants/{field}"],
                    inline_array=inline_array,
                    chunks=chunks,
                ),
            )

        # call arrays
        data_vars["call_genotype"] = (
            [DIM_VARIANT, DIM_SAMPLE],
            da_from_zarr(
                root[f"{contig}/calldata/GT"], inline_array=inline_array, chunks=chunks
            ),
        )

        # sample arrays
        coords["sample_id"] = (
            [DIM_SAMPLE],
            da_from_zarr(root["samples"], inline_array=inline_array, chunks=chunks),
        )
        for field in "sample_coverage_variance", "sample_is_high_variance":
            data_vars[field] = (
                [DIM_SAMPLE],
                da_from_zarr(root[field], inline_array=inline_array, chunks=chunks),
            )

        # setup attributes
        attrs = {"contigs": self.contigs}

        # create a dataset
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        return ds

    def cnv_discordant_read_calls(
        self,
        contig,
        sample_sets=None,
        inline_array=True,
        chunks="native",
    ):
        """Access CNV discordant read calls data.

        Parameters
        ----------
        contig : str
            Chromosome arm, e.g., "3R". Multiple values can be provided
            as a list, in which case data will be concatenated, e.g., ["2R", "3R"].
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "3.0") or a list of release identifiers.
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr chunks.
            Also can be a target size, e.g., '200 MiB'.

        Returns
        -------
        ds : xarray.Dataset
            A dataset of CNV alleles and genotypes.

        """

        # normalise parameters
        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)
        if isinstance(contig, str):
            contig = [contig]

        # concatenate
        lx = []
        for c in contig:

            ly = []
            for s in sample_sets:
                y = self._cnv_discordant_read_calls_dataset(
                    contig=c,
                    sample_set=s,
                    inline_array=inline_array,
                    chunks=chunks,
                )
                ly.append(y)

            x = xarray_concat(ly, dim=DIM_SAMPLE)
            lx.append(x)

        ds = xarray_concat(lx, dim=DIM_VARIANT)

        return ds

    def gene_cnv(self, contig, sample_sets=None):
        """Compute modal copy number by gene, from HMM data.

        Parameters
        ----------
        contig : str
            Chromosome arm, e.g., "3R".
        sample_sets : str or list of str
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "3.0") or a list of release identifiers.

        Returns
        -------
        ds : xarray.Dataset
            A dataset of modal copy number per gene and associated data.

        """

        # access HMM data
        ds_hmm = self.cnv_hmm(contig=contig, sample_sets=sample_sets)
        pos = ds_hmm["variant_position"].values
        end = ds_hmm["variant_end"].values
        cn = ds_hmm["call_CN"].values

        # access genes
        df_geneset = self.geneset()
        df_genes = df_geneset.query(f"type == 'gene' and contig == '{contig}'")

        # setup intermediates
        windows = []
        modes = []
        counts = []

        # iterate over genes
        for gene in df_genes.itertuples():

            # locate windows overlapping the gene
            loc_gene_start = bisect_left(end, gene.start)
            loc_gene_stop = bisect_right(pos, gene.end)
            w = loc_gene_stop - loc_gene_start
            windows.append(w)

            # slice out copy number data for the given gene
            cn_gene = cn[loc_gene_start:loc_gene_stop]

            # compute the modes
            m, c = _cn_mode(cn_gene, vmax=12)
            modes.append(m)
            counts.append(c)

        # combine results
        windows = np.array(windows)
        modes = np.vstack(modes)
        counts = np.vstack(counts)

        # build dataset
        ds_out = xr.Dataset(
            coords={
                "gene_id": (["genes"], df_genes["ID"].values),
                "gene_start": (["genes"], df_genes["start"].values),
                "gene_end": (["genes"], df_genes["end"].values),
                "sample_id": (["samples"], ds_hmm["sample_id"].values),
            },
            data_vars={
                "gene_windows": (["genes"], windows),
                "gene_name": (["genes"], df_genes["Name"].values),
                "gene_strand": (["genes"], df_genes["strand"].values),
                "CN_mode": (["genes", "samples"], modes),
                "CN_mode_count": (["genes", "samples"], counts),
                "sample_coverage_variance": (
                    ["samples"],
                    ds_hmm["sample_coverage_variance"].values,
                ),
                "sample_is_high_variance": (
                    ["samples"],
                    ds_hmm["sample_is_high_variance"].values,
                ),
            },
        )

        return ds_out

    def gene_cnv_frequencies(
        self,
        contig,
        cohorts,
        sample_query=None,
        cohorts_analysis=DEFAULT_COHORTS_ANALYSIS,
        min_cohort_size=10,
        species_analysis=DEFAULT_SPECIES_ANALYSIS,
        sample_sets=None,
        drop_invariant=True,
        max_coverage_variance=0.2,
    ):
        """Compute modal copy number by gene, then compute the frequency of
        amplifications and deletions in one or more cohorts, from HMM data.

        Parameters
        ----------
        contig : str
            Chromosome arm, e.g., "3R".
        cohorts : str or dict
            If a string, gives the name of a predefined cohort set, e.g., one of
            {"admin1_month", "admin1_year", "admin2_month", "admin2_year"}.
            If a dict, should map cohort labels to sample queries, e.g.,
            `{"bf_2012_col": "country == 'Burkina Faso' and year == 2012 and
            taxon == 'coluzzii'"}`.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        cohorts_analysis : str
            Cohort analysis identifier (date of analysis), default is the latest
            version.
        min_cohort_size : int
            Minimum cohort size, below which cohorts are dropped.
        species_analysis : {"aim_20200422", "pca_20200422"}, optional
            Include species calls in metadata.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        drop_invariant : bool, optional
            If True, drop any rows where there is no evidence of variation.
        max_coverage_variance : float, optional
            Remove samples if coverage variance exceeds this value.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of CNV amplification (amp) and deletion (del) frequencies in the specified cohorts,
            one row per gene and CNV type (amp/del).

        """

        # check parameters
        _check_param_min_cohort_size(min_cohort_size)

        # load sample metadata
        df_samples = self.sample_metadata(
            sample_sets=sample_sets,
            cohorts_analysis=cohorts_analysis,
            species_analysis=species_analysis,
        )

        # get gene copy number data
        ds_cnv = self.gene_cnv(contig=contig, sample_sets=sample_sets)

        # handle sample_query
        loc_samples = None
        if sample_query is not None:
            loc_samples = df_samples.eval(sample_query).values

        # handle filtering samples by coverage variance
        if max_coverage_variance is not None:
            cov_var = ds_cnv["sample_coverage_variance"].values
            loc_pass_samples = cov_var <= max_coverage_variance
            if loc_samples is not None:
                loc_samples = loc_samples & loc_pass_samples
            else:
                loc_samples = loc_pass_samples

        # get genes
        df_genes = self.geneset().query(f"type == 'gene' and contig == '{contig}'")

        # figure out expected copy number
        if contig == "X":
            is_male = (df_samples["sex_call"] == "M").values
            expected_cn = np.where(is_male, 1, 2)[np.newaxis, :]
        else:
            expected_cn = 2

        # setup output dataframe
        n_genes = len(df_genes)
        df = pd.concat([df_genes, df_genes], axis=0).reset_index(drop=True)

        # drop columns we don't need
        df.drop(columns=["source", "type", "score", "phase", "Parent"], inplace=True)

        # rename some columns
        df.rename(
            columns={
                "ID": "gene_id",
                "Name": "gene_name",
                "strand": "gene_strand",
                "description": "gene_description",
            },
            inplace=True,
        )

        # add CNV type column
        df_cnv_type = pd.DataFrame(
            {
                "cnv_type": np.array(
                    (["amp"] * n_genes) + (["del"] * n_genes), dtype=object
                )
            }
        )
        df = pd.concat([df, df_cnv_type], axis=1)

        # set up intermediates
        cn = ds_cnv["CN_mode"].values
        is_amp = cn > expected_cn
        is_del = (cn >= 0) & (cn < expected_cn)

        # set up cohort dict
        coh_dict = self._prep_cohorts_arg(
            cohorts=cohorts,
            sample_sets=sample_sets,
            species_analysis=species_analysis,
            cohorts_analysis=cohorts_analysis,
        )

        # compute cohort frequencies
        freq_cols = dict()
        for coh, loc_coh in coh_dict.items():
            # handle sample query
            if loc_samples is not None:
                loc_coh = loc_coh & loc_samples
            n_samples = np.count_nonzero(loc_coh)
            if n_samples >= min_cohort_size:
                is_amp_coh = np.compress(loc_coh, is_amp, axis=1)
                is_del_coh = np.compress(loc_coh, is_del, axis=1)
                amp_count_coh = np.sum(is_amp_coh, axis=1)
                del_count_coh = np.sum(is_del_coh, axis=1)
                amp_freq_coh = amp_count_coh / n_samples
                del_freq_coh = del_count_coh / n_samples
                freq_cols[f"frq_{coh}"] = np.concatenate([amp_freq_coh, del_freq_coh])

        # build a dataframe with the frequency columns
        df_freqs = pd.DataFrame(freq_cols)

        # compute max_af and additional columns
        df_extras = pd.DataFrame(
            {
                "max_af": df_freqs.max(axis=1),
                "windows": np.concatenate(
                    [ds_cnv["gene_windows"].values, ds_cnv["gene_windows"].values]
                ),
            }
        )

        # build the final dataframe
        df.reset_index(drop=True, inplace=True)
        df = pd.concat([df, df_freqs, df_extras], axis=1)
        df.sort_values(["contig", "start", "cnv_type"], inplace=True)

        # add label
        df["label"] = df.apply(_make_gene_cnv_label, axis="columns")

        # deal with invariants
        if drop_invariant:
            df = df.query("max_af > 0")

        # set index for convenience
        df.set_index(["gene_id", "gene_name", "cnv_type"], inplace=True)

        # add metadata
        title = "Gene CNV frequencies"
        df.attrs["title"] = title

        return df

    def open_haplotypes(self, sample_set, analysis):
        """Open haplotypes zarr.

        Parameters
        ----------
        sample_set : str
            Sample set identifier, e.g., "AG1000G-AO".
        analysis : {"arab", "gamb_colu", "gamb_colu_arab"}
            Which phasing analysis to use. If analysing only An. arabiensis, the "arab" analysis
            is best. If analysing only An. gambiae and An. coluzzii, the "gamb_colu" analysis is
            best. Otherwise, use the "gamb_colu_arab" analysis.

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        try:
            return self._cache_haplotypes[(sample_set, analysis)]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            release_path = _release_to_path(release)
            path = f"{self._base_path}/{release_path}/snp_haplotypes/{sample_set}/{analysis}/zarr"
            store = init_zarr_store(fs=self._fs, path=path)
            # some sample sets have no data for a given analysis, handle this
            try:
                root = zarr.open_consolidated(store=store)
            except FileNotFoundError:
                root = None
            self._cache_haplotypes[(sample_set, analysis)] = root
        return root

    def open_haplotype_sites(self, analysis):
        """Open haplotype sites zarr.

        Parameters
        ----------
        analysis : {"arab", "gamb_colu", "gamb_colu_arab"}
            Which phasing analysis to use. If analysing only An. arabiensis, the "arab" analysis
            is best. If analysing only An. gambiae and An. coluzzii, the "gamb_colu" analysis is
            best. Otherwise, use the "gamb_colu_arab" analysis.

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        try:
            return self._cache_haplotype_sites[analysis]
        except KeyError:
            path = f"{self._base_path}/v3/snp_haplotypes/sites/{analysis}/zarr"
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_haplotype_sites[analysis] = root
        return root

    def _haplotypes_dataset(
        self, *, contig, sample_set, analysis, inline_array, chunks
    ):

        # open zarr
        root = self.open_haplotypes(sample_set=sample_set, analysis=analysis)
        sites = self.open_haplotype_sites(analysis=analysis)

        # some sample sets have no data for a given analysis, handle this
        # TODO consider returning a dataset with 0 length samples dimension instead, would
        # probably simplify a lot of other logic
        if root is None:
            return None

        coords = dict()
        data_vars = dict()

        # variant_position
        pos = sites[f"{contig}/variants/POS"]
        coords["variant_position"] = (
            [DIM_VARIANT],
            da_from_zarr(pos, inline_array=inline_array, chunks=chunks),
        )

        # variant_contig
        contig_index = self.contigs.index(contig)
        coords["variant_contig"] = (
            [DIM_VARIANT],
            da.full_like(pos, fill_value=contig_index, dtype="u1"),
        )

        # variant_allele
        ref = da_from_zarr(
            sites[f"{contig}/variants/REF"], inline_array=inline_array, chunks=chunks
        )
        alt = da_from_zarr(
            sites[f"{contig}/variants/ALT"], inline_array=inline_array, chunks=chunks
        )
        variant_allele = da.hstack([ref[:, None], alt[:, None]])
        data_vars["variant_allele"] = [DIM_VARIANT, DIM_ALLELE], variant_allele

        # call_genotype
        data_vars["call_genotype"] = (
            [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY],
            da_from_zarr(
                root[f"{contig}/calldata/GT"], inline_array=inline_array, chunks=chunks
            ),
        )

        # sample arrays
        coords["sample_id"] = (
            [DIM_SAMPLE],
            da_from_zarr(root["samples"], inline_array=inline_array, chunks=chunks),
        )

        # setup attributes
        attrs = {"contigs": self.contigs}

        # create a dataset
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        return ds

    def haplotypes(
        self,
        region,
        analysis,
        sample_sets=None,
        inline_array=True,
        chunks="native",
    ):
        """Access haplotype data.

        Parameters
        ----------
        region: str or list of str or Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic region
            defined with coordinates (e.g., "2L:44989425-44998059") or a named tuple with
            genomic location `Region(contig, start, end)`. Multiple values can be provided
            as a list, in which case data will be concatenated, e.g., ["3R", "AGAP005958"].
        analysis : {"arab", "gamb_colu", "gamb_colu_arab"}
            Which phasing analysis to use. If analysing only An. arabiensis, the "arab" analysis
            is best. If analysing only An. gambiae and An. coluzzii, the "gamb_colu" analysis is
            best. Otherwise use the "gamb_colu_arab" analysis.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "3.0") or a list of release identifiers.
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr chunks.
            Also can be a target size, e.g., '200 MiB'.

        Returns
        -------
        ds : xarray.Dataset
            A dataset of haplotypes and associated data.

        """

        # normalise parameters
        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)
        region = resolve_region(self, region)

        # normalise to simplify concatenation logic
        if isinstance(region, Region):
            region = [region]

        # build dataset
        lx = []
        for r in region:
            ly = []

            for s in sample_sets:
                y = self._haplotypes_dataset(
                    contig=r.contig,
                    sample_set=s,
                    analysis=analysis,
                    inline_array=inline_array,
                    chunks=chunks,
                )
                if y is not None:
                    ly.append(y)

            if len(ly) == 0:
                # early out, no data for given sample sets and analysis
                return None

            # concatenate data from multiple sample sets
            x = xarray_concat(ly, dim=DIM_SAMPLE)

            # handle region
            if r.start or r.end:
                pos = x["variant_position"].values
                loc_region = locate_region(r, pos)
                x = x.isel(variants=loc_region)

            lx.append(x)

        # concatenate data from multiple regions
        ds = xarray_concat(lx, dim=DIM_VARIANT)

        return ds

    def _read_cohort_metadata(self, *, sample_set, cohorts_analysis):
        """Read cohort metadata for a single sample set."""
        try:
            return self._cache_cohort_metadata[(sample_set, cohorts_analysis)]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            release_path = _release_to_path(release)
            path_prefix = f"{self._base_path}/{release_path}/metadata"
            path = f"{path_prefix}/cohorts_{cohorts_analysis}/{sample_set}/samples.cohorts.csv"
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

            self._cache_cohort_metadata[(sample_set, cohorts_analysis)] = df
            return df

    def sample_cohorts(
        self, sample_sets=None, cohorts_analysis=DEFAULT_COHORTS_ANALYSIS
    ):
        """Access cohorts metadata for one or more sample sets.

        Parameters
        ----------
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "3.0") or a list of release identifiers.
        cohorts_analysis : str
            Cohort analysis identifier (date of analysis), default is the latest version.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of cohort metadata, one row per sample.

        """
        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)

        # concatenate multiple sample sets
        dfs = [
            self._read_cohort_metadata(sample_set=s, cohorts_analysis=cohorts_analysis)
            for s in sample_sets
        ]
        df = pd.concat(dfs, axis=0, ignore_index=True)

        return df

    def aa_allele_frequencies(
        self,
        transcript,
        cohorts,
        sample_query=None,
        cohorts_analysis=DEFAULT_COHORTS_ANALYSIS,
        min_cohort_size=10,
        site_mask=None,
        site_filters_analysis=DEFAULT_SITE_FILTERS_ANALYSIS,
        species_analysis=DEFAULT_SPECIES_ANALYSIS,
        sample_sets=None,
        drop_invariant=True,
    ):
        """Compute per amino acid allele frequencies for a gene transcript.

        Parameters
        ----------
        transcript : str
            Gene transcript ID (AgamP4.12), e.g., "AGAP004707-RA".
        cohorts : str or dict
            If a string, gives the name of a predefined cohort set, e.g., one of
            {"admin1_month", "admin1_year", "admin2_month", "admin2_year"}.
            If a dict, should map cohort labels to sample queries, e.g.,
            `{"bf_2012_col": "country == 'Burkina Faso' and year == 2012 and taxon == 'coluzzii'"}`.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample metadata e.g.,
            "taxon == 'coluzzii' and country == 'Burkina Faso'".
        cohorts_analysis : str
            Cohort analysis identifier (date of analysis), default is latest version.
        min_cohort_size : int
            Minimum cohort size, below which allele frequencies are not calculated for cohorts.
            Please note, NaNs will be returned for any cohorts with fewer samples than min_cohort_size,
            these can be removed from the output dataframe using pandas df.dropna(axis='columns').
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_filters_analysis : str, optional
            Site filters analysis version.
        species_analysis : {"aim_20200422", "pca_20200422"}, optional
            Include species calls in metadata.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "3.0") or a list of release identifiers.
        drop_invariant : bool, optional
            If True, variants with no alternate allele calls in any cohorts are dropped from
            the result.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of amino acid allele frequencies, one row per replacement.

        Notes
        -----
        Cohorts with fewer samples than min_cohort_size will be excluded from output.

        """

        df_snps = self.snp_allele_frequencies(
            transcript=transcript,
            cohorts=cohorts,
            sample_query=sample_query,
            cohorts_analysis=cohorts_analysis,
            min_cohort_size=min_cohort_size,
            site_mask=site_mask,
            site_filters_analysis=site_filters_analysis,
            species_analysis=species_analysis,
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

        # group and sum to collapse multi variant allele changes
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

        # compute new max_af
        df_aaf["max_af"] = df_aaf[freq_cols].max(axis=1)

        # add label
        df_aaf["label"] = df_aaf.apply(_make_snp_label_aa, axis="columns")

        # sort by genomic position
        df_aaf = df_aaf.sort_values(["position", "aa_change"])

        # set index
        df_aaf.set_index(["aa_change", "contig", "position"], inplace=True)

        # add metadata
        gene_name = self._transcript_to_gene_name(transcript)
        title = transcript
        if gene_name:
            title += f" ({gene_name})"
        title += " SNP frequencies"
        df_aaf.attrs["title"] = title

        return df_aaf

    @staticmethod
    def plot_frequencies_heatmap(
        df,
        index="label",
        max_len=100,
        x_label="cohorts",
        y_label="variants",
        colorbar=True,
        width=None,
        height=None,
        text_auto=".0%",
        aspect="auto",
        color_continuous_scale="Reds",
        title=True,
        **kwargs,
    ):

        """Plot a heatmap from a pandas DataFrame of frequencies, e.g., output from
            `Ag3.snp_allele_frequencies()` or `Ag3.gene_cnv_frequencies()`. It's recommended to
            filter the input DataFrame to just rows of interest, i.e., fewer rows than `max_len`.

        Parameters
        ----------
        df : pandas DataFrame
           A DataFrame of frequencies, e.g., output from `snp_allele_frequencies()` or `gene_cnv_frequencies()`.
        index : str or list of str
            One or more column headers that are present in the input dataframe. This becomes the heatmap y-axis
            row labels. The column/s must produce a unique index.
        max_len : int, optional
            Displaying large styled dataframes may cause ipython notebooks to crash.
        x_label : str, optional
            This is the x-axis label that will be displayed on the heatmap.
        y_label : str, optional
            This is the y-axis label that will be displayed on the heatmap.
        colorbar : bool, optional
            If False, colorbar is not output.
        width : int, optional
            Plot width in pixels.
        height : int, optional
            Plot height in pixels.
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

        """

        # check len of input
        if len(df) > max_len:
            raise ValueError(f"Input DataFrame is longer than {max_len}")

        # handle title
        if title is True:
            title = df.attrs.get("title", None)

        # indexing
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

        # check that index is unique
        if not index_col.is_unique:
            raise ValueError(f"{index} does not produce a unique index")

        # drop and re-order columns
        frq_cols = [col for col in df.columns if col.startswith("frq_")]

        # keep only freq cols
        heatmap_df = df[frq_cols].copy()

        # set index
        heatmap_df.set_index(index_col, inplace=True)

        # clean column names
        heatmap_df.columns = heatmap_df.columns.str.lstrip("frq_")

        # plotly heatmap styling
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
                title="frequency",
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["0%", "20%", "40%", "60%", "80%", "100%"],
            )
        )
        if not colorbar:
            fig.update(layout_coloraxis_showscale=False)

        return fig

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
        cohorts_analysis=DEFAULT_COHORTS_ANALYSIS,
        species_analysis=DEFAULT_SPECIES_ANALYSIS,
        site_filters_analysis=DEFAULT_SITE_FILTERS_ANALYSIS,
    ):
        """Group samples by taxon, area (space) and period (time), then compute SNP allele counts
        and frequencies.

        Parameters
        ----------
        transcript : str
            Gene transcript ID (AgamP4.12), e.g., "AGAP004707-RD".
        area_by : str
            Column name in the sample metadata to use to group samples spatially. E.g.,
            use "adm1_ISO" or "adm1_name" to group by level 1 administrative divisions,
            or use "adm2_name" to group by level 2 administrative divisions.
        period_by : {"year", "quarter", "month"}
            Length of time to group samples temporally.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample metadata e.g.,
            "taxon == 'coluzzii' and country == 'Burkina Faso'".
        min_cohort_size : int, optional
            Minimum cohort size. Any cohorts below this size are omitted.
        drop_invariant : bool, optional
            If True, variants with no alternate allele calls in any cohorts are dropped from
            the result.
        variant_query : str, optional
        site_mask : str, optional
            Site filters mask to apply.
        nobs_mode : {"called", "fixed"}
            Method for calculating the denominator when computing frequencies. If "called"
            then use the number of called alleles, i.e., number of samples with non-missing
            genotype calls multiplied by 2. If "fixed" then use the number of samples
            multiplied by 2.
        ci_method : {"normal", "agresti_coull", "beta", "wilson", "binom_test"}, optional
            Method to use for computing confidence intervals, passed through to
            `statsmodels.stats.proportion.proportion_confint`.
        cohorts_analysis : str, optional
            Cohort analysis version, default is the latest version.
        species_analysis : str, optional
            Species calls analysis version.
        site_filters_analysis : str, optional
            Site filters analysis version.

        Returns
        -------
        ds : xarray.Dataset
            The resulting dataset contains data has dimensions "cohorts" and "variants".
            Variables prefixed with "cohort_" are 1-dimensional arrays with data about
            the cohorts, such as the area, period, taxon and cohort size. Variables
            prefixed with "variant_" are 1-dimensional arrays with data about the variants,
            such as the contig, position, reference and alternate alleles. Variables prefixed
            with "event_" are 2-dimensional arrays with the allele counts and frequency
            calculations.

        """

        # check parameters
        _check_param_min_cohort_size(min_cohort_size)

        # load sample metadata
        df_samples = self.sample_metadata(
            sample_sets=sample_sets,
            species_analysis=species_analysis,
            cohorts_analysis=cohorts_analysis,
        )

        # access SNP calls
        ds_snps = self.snp_calls(
            region=transcript,
            sample_sets=sample_sets,
            site_mask=site_mask,
            site_filters_analysis=site_filters_analysis,
        )

        # access genotypes
        gt = ds_snps["call_genotype"].data

        # handle sample query
        loc_samples = None
        if sample_query is not None:
            loc_samples = df_samples.eval(sample_query).values

        # filter samples
        if loc_samples is not None:
            df_samples = df_samples.loc[loc_samples].reset_index(drop=True).copy()
            gt = da.compress(loc_samples, gt, axis=1)

        # prepare sample metadata for cohort grouping
        df_samples = _prep_samples_for_cohort_grouping(
            df_samples=df_samples,
            area_by=area_by,
            period_by=period_by,
        )

        # group samples to make cohorts
        group_samples_by_cohort = df_samples.groupby(["taxon", "area", "period"])

        # build cohorts dataframe
        df_cohorts = _build_cohorts_from_sample_grouping(
            group_samples_by_cohort, min_cohort_size
        )

        # bring genotypes into memory
        gt = gt.compute()

        # convert genotypes to more convenient representation
        gac, gan = _genotypes_to_alt_allele_counts_melt(gt, max_allele=3)

        # set up variant variables
        contigs = ds_snps.attrs["contigs"]
        variant_contig = np.repeat(
            [contigs[i] for i in ds_snps["variant_contig"].values], 3
        )
        variant_position = np.repeat(ds_snps["variant_position"].values, 3)
        alleles = ds_snps["variant_allele"].values
        variant_ref_allele = np.repeat(alleles[:, 0], 3)
        variant_alt_allele = alleles[:, 1:].flatten()
        variant_pass_gamb_colu_arab = np.repeat(
            ds_snps["variant_filter_pass_gamb_colu_arab"].values, 3
        )
        variant_pass_gamb_colu = np.repeat(
            ds_snps["variant_filter_pass_gamb_colu"].values, 3
        )
        variant_pass_arab = np.repeat(ds_snps["variant_filter_pass_arab"].values, 3)

        # setup main event variables
        n_variants, n_cohorts = len(variant_position), len(df_cohorts)
        count = np.zeros((n_variants, n_cohorts), dtype=int)
        nobs = np.zeros((n_variants, n_cohorts), dtype=int)

        # build event count and nobs for each cohort
        for cohort_index, cohort in enumerate(df_cohorts.itertuples()):

            # construct grouping key
            cohort_key = cohort.taxon, cohort.area, cohort.period

            # obtain sample indices for cohort
            sample_indices = group_samples_by_cohort.indices[cohort_key]

            # compute cohort allele counts
            # cohort_gac = np.take(gac, sample_indices, axis=1)
            # np.sum(cohort_gac, axis=1, out=count[:, cohort_index])
            count[:, cohort_index] = _take_sum_cols(gac, sample_indices)

            # compute cohort allele numbers
            if nobs_mode == "called":
                # cohort_gan = np.take(gan, sample_indices, axis=1)
                # np.sum(cohort_gan, axis=1, out=nobs[:, cohort_index])
                nobs[:, cohort_index] = _take_sum_cols(gan, sample_indices)
            elif nobs_mode == "fixed":
                nobs[:, cohort_index] = cohort.size * 2
            else:
                raise ValueError(f"Bad nobs_mode: {nobs_mode!r}")

        # compute frequency
        with np.errstate(divide="ignore", invalid="ignore"):
            # ignore division warnings
            frequency = count / nobs

        # compute maximum frequency over cohorts
        with warnings.catch_warnings():
            # ignore "All-NaN slice encountered" warnings
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_af = np.nanmax(frequency, axis=1)

        # make dataframe of SNPs
        df_variants = pd.DataFrame(
            {
                "contig": variant_contig,
                "position": variant_position,
                "ref_allele": variant_ref_allele.astype("U1"),
                "alt_allele": variant_alt_allele.astype("U1"),
                "max_af": max_af,
                "pass_gamb_colu_arab": variant_pass_gamb_colu_arab,
                "pass_gamb_colu": variant_pass_gamb_colu,
                "pass_arab": variant_pass_arab,
            }
        )

        # deal with SNP alleles not observed
        if drop_invariant:
            loc_variant = max_af > 0
            df_variants = df_variants.loc[loc_variant]
            count = np.compress(loc_variant, count, axis=0)
            nobs = np.compress(loc_variant, nobs, axis=0)
            frequency = np.compress(loc_variant, frequency, axis=0)

        # setup variant effect annotator
        ann = self._annotator()

        # add effects to the dataframe
        ann.get_effects(transcript=transcript, variants=df_variants)

        # add variant labels
        df_variants["label"] = df_variants.apply(_make_snp_label_effect, axis="columns")

        # build the output dataset
        ds_out = xr.Dataset()

        # cohort variables
        for coh_col in df_cohorts.columns:
            ds_out[f"cohort_{coh_col}"] = "cohorts", df_cohorts[coh_col]

        # variant variables
        for snp_col in df_variants.columns:
            ds_out[f"variant_{snp_col}"] = "variants", df_variants[snp_col]

        # event variables
        ds_out["event_count"] = ("variants", "cohorts"), count
        ds_out["event_nobs"] = ("variants", "cohorts"), nobs
        ds_out["event_frequency"] = ("variants", "cohorts"), frequency

        # apply variant query
        if variant_query is not None:
            loc_variants = df_variants.eval(variant_query).values
            ds_out = ds_out.isel(variants=loc_variants)

        # add confidence intervals
        _add_frequency_ci(ds_out, ci_method)

        # tidy up display by sorting variables
        ds_out = ds_out[sorted(ds_out)]

        # add metadata
        gene_name = self._transcript_to_gene_name(transcript)
        title = transcript
        if gene_name:
            title += f" ({gene_name})"
        title += " SNP frequencies"
        ds_out.attrs["title"] = title

        return ds_out

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
        cohorts_analysis=DEFAULT_COHORTS_ANALYSIS,
        species_analysis=DEFAULT_SPECIES_ANALYSIS,
        site_filters_analysis=DEFAULT_SITE_FILTERS_ANALYSIS,
    ):
        """Group samples by taxon, area (space) and period (time), then compute amino acid
        change allele counts and frequencies.

        Parameters
        ----------
        transcript : str
            Gene transcript ID (AgamP4.12), e.g., "AGAP004707-RD".
        area_by : str
            Column name in the sample metadata to use to group samples spatially. E.g.,
            use "adm1_ISO" or "adm1_name" to group by level 1 administrative divisions,
            or use "adm2_name" to group by level 2 administrative divisions.
        period_by : {"year", "quarter", "month"}
            Length of time to group samples temporally.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample metadata e.g.,
            "taxon == 'coluzzii' and country == 'Burkina Faso'".
        min_cohort_size : int, optional
            Minimum cohort size. Any cohorts below this size are omitted.
        variant_query : str, optional
        site_mask : str, optional
            Site filters mask to apply.
        nobs_mode : {"called", "fixed"}
            Method for calculating the denominator when computing frequencies. If "called"
            then use the number of called alleles, i.e., number of samples with non-missing
            genotype calls multiplied by 2. If "fixed" then use the number of samples
            multiplied by 2.
        ci_method : {"normal", "agresti_coull", "beta", "wilson", "binom_test"}, optional
            Method to use for computing confidence intervals, passed through to
            `statsmodels.stats.proportion.proportion_confint`.
        cohorts_analysis : str, optional
            Cohort analysis version, default is the latest version.
        species_analysis : str, optional
            Species calls analysis version.
        site_filters_analysis : str, optional
            Site filters analysis version.

        Returns
        -------
        ds : xarray.Dataset
            The resulting dataset contains data has dimensions "cohorts" and "variants".
            Variables prefixed with "cohort_" are 1-dimensional arrays with data about
            the cohorts, such as the area, period, taxon and cohort size. Variables
            prefixed with "variant_" are 1-dimensional arrays with data about the variants,
            such as the contig, position, reference and alternate alleles. Variables prefixed
            with "event_" are 2-dimensional arrays with the allele counts and frequency
            calculations.

        """

        # begin by computing SNP allele frequencies
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
            cohorts_analysis=cohorts_analysis,
            species_analysis=species_analysis,
            site_filters_analysis=site_filters_analysis,
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

        # group by position and amino acid change
        group_by_aa_change = ds_snp_frq.groupby("variant_position_aa_change")

        # apply aggregation
        ds_aa_frq = group_by_aa_change.map(_map_snp_to_aa_change_frq_ds)

        # add back in cohort variables, unaffected by aggregation
        cohort_vars = [v for v in ds_snp_frq if v.startswith("cohort_")]
        for v in cohort_vars:
            ds_aa_frq[v] = ds_snp_frq[v]

        # sort by genomic position
        ds_aa_frq = ds_aa_frq.sortby(["variant_position", "variant_aa_change"])

        # recompute frequency
        count = ds_aa_frq["event_count"].values
        nobs = ds_aa_frq["event_nobs"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            # ignore division warnings
            frequency = count / nobs
        ds_aa_frq["event_frequency"] = ("variants", "cohorts"), frequency

        # recompute max frequency over cohorts
        with warnings.catch_warnings():
            # ignore "All-NaN slice encountered" warnings
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_af = np.nanmax(ds_aa_frq["event_frequency"].values, axis=1)
        ds_aa_frq["variant_max_af"] = "variants", max_af

        # set up variant dataframe, useful intermediate
        variant_cols = [v for v in ds_aa_frq if v.startswith("variant_")]
        df_variants = ds_aa_frq[variant_cols].to_dataframe()
        df_variants.columns = [c.split("variant_")[1] for c in df_variants.columns]

        # assign new variant label
        label = df_variants.apply(_make_snp_label_aa, axis=1)
        ds_aa_frq["variant_label"] = "variants", label

        # apply variant query if given
        if variant_query is not None:
            loc_variants = df_variants.eval(variant_query).values
            ds_aa_frq = ds_aa_frq.isel(variants=loc_variants)
            # df_variants = df_variants.loc[loc_variants]

        # compute new confidence intervals
        _add_frequency_ci(ds_aa_frq, ci_method)

        # tidy up display by sorting variables
        ds_aa_frq = ds_aa_frq[sorted(ds_aa_frq)]

        gene_name = self._transcript_to_gene_name(transcript)
        title = transcript
        if gene_name:
            title += f" ({gene_name})"
        title += " SNP frequencies"
        ds_aa_frq.attrs["title"] = title

        return ds_aa_frq

    def gene_cnv_frequencies_advanced(
        self,
        contig,
        area_by,
        period_by,
        sample_sets=None,
        sample_query=None,
        min_cohort_size=10,
        variant_query=None,
        drop_invariant=True,
        max_coverage_variance=0.2,
        ci_method="wilson",
        cohorts_analysis=DEFAULT_COHORTS_ANALYSIS,
        species_analysis=DEFAULT_SPECIES_ANALYSIS,
    ):
        """Group samples by taxon, area (space) and period (time), then compute gene CNV counts
        and frequencies.

        Parameters
        ----------
        contig : str
            Chromosome arm, e.g., "3R".
        area_by : str
            Column name in the sample metadata to use to group samples spatially. E.g.,
            use "adm1_ISO" or "adm1_name" to group by level 1 administrative divisions,
            or use "adm2_name" to group by level 2 administrative divisions.
        period_by : {"year", "quarter", "month"}
            Length of time to group samples temporally.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample metadata e.g.,
            "taxon == 'coluzzii' and country == 'Burkina Faso'".
        min_cohort_size : int, optional
            Minimum cohort size. Any cohorts below this size are omitted.
        variant_query : str, optional
        drop_invariant : bool, optional
            If True, drop any rows where there is no evidence of variation.
        max_coverage_variance : float, optional
            Remove samples if coverage variance exceeds this value.
        ci_method : {"normal", "agresti_coull", "beta", "wilson", "binom_test"}, optional
            Method to use for computing confidence intervals, passed through to
            `statsmodels.stats.proportion.proportion_confint`.
        cohorts_analysis : str, optional
            Cohort analysis version, default is the latest version.
        species_analysis : str, optional
            Species calls analysis version.

        Returns
        -------
        ds : xarray.Dataset
            The resulting dataset contains data has dimensions "cohorts" and "variants".
            Variables prefixed with "cohort_" are 1-dimensional arrays with data about
            the cohorts, such as the area, period, taxon and cohort size. Variables
            prefixed with "variant_" are 1-dimensional arrays with data about the variants,
            such as the contig, position, reference and alternate alleles. Variables prefixed
            with "event_" are 2-dimensional arrays with the allele counts and frequency
            calculations.

        """

        # check parameters
        _check_param_min_cohort_size(min_cohort_size)

        # load sample metadata
        df_samples = self.sample_metadata(
            sample_sets=sample_sets,
            species_analysis=species_analysis,
            cohorts_analysis=cohorts_analysis,
        )

        # access gene CNV calls
        ds_cnv = self.gene_cnv(contig=contig, sample_sets=sample_sets)

        # handle sample query
        loc_samples = None
        if sample_query is not None:
            loc_samples = df_samples.eval(sample_query).values

        # handle filtering samples by coverage variance
        if max_coverage_variance is not None:
            cov_var = ds_cnv["sample_coverage_variance"].values
            loc_pass_samples = cov_var <= max_coverage_variance
            if loc_samples is not None:
                loc_samples = loc_samples & loc_pass_samples
            else:
                loc_samples = loc_pass_samples

        # filter samples
        if loc_samples is not None:
            df_samples = df_samples.loc[loc_samples].reset_index(drop=True).copy()
            ds_cnv = ds_cnv.isel(samples=loc_samples)

        # prepare sample metadata for cohort grouping
        df_samples = _prep_samples_for_cohort_grouping(
            df_samples=df_samples,
            area_by=area_by,
            period_by=period_by,
        )

        # group samples to make cohorts
        group_samples_by_cohort = df_samples.groupby(["taxon", "area", "period"])

        # build cohorts dataframe
        df_cohorts = _build_cohorts_from_sample_grouping(
            group_samples_by_cohort, min_cohort_size
        )

        # figure out expected copy number
        if contig == "X":
            is_male = (df_samples["sex_call"] == "M").values
            expected_cn = np.where(is_male, 1, 2)[np.newaxis, :]
        else:
            expected_cn = 2

        # set up intermediates
        cn = ds_cnv["CN_mode"].values
        is_amp = cn > expected_cn
        is_del = (cn >= 0) & (cn < expected_cn)

        # set up main event variables
        n_genes = ds_cnv.dims["genes"]
        n_variants, n_cohorts = n_genes * 2, len(df_cohorts)
        count = np.zeros((n_variants, n_cohorts), dtype=int)
        nobs = np.zeros((n_variants, n_cohorts), dtype=int)

        # build event count and nobs for each cohort
        for cohort_index, cohort in enumerate(df_cohorts.itertuples()):

            # construct grouping key
            cohort_key = cohort.taxon, cohort.area, cohort.period

            # obtain sample indices for cohort
            sample_indices = group_samples_by_cohort.indices[cohort_key]

            # select genotype data for cohort
            cohort_is_amp = np.take(is_amp, sample_indices, axis=1)
            cohort_is_del = np.take(is_del, sample_indices, axis=1)

            # compute cohort allele counts
            np.sum(cohort_is_amp, axis=1, out=count[::2, cohort_index])
            np.sum(cohort_is_del, axis=1, out=count[1::2, cohort_index])

            # compute cohort allele numbers
            nobs[:, cohort_index] = cohort.size

        # compute frequency
        with np.errstate(divide="ignore", invalid="ignore"):
            # ignore division warnings
            frequency = count / nobs

        # make dataframe of variants
        with warnings.catch_warnings():
            # ignore "All-NaN slice encountered" warnings
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_af = np.nanmax(frequency, axis=1)
        df_variants = pd.DataFrame(
            {
                "contig": contig,
                "start": np.repeat(ds_cnv["gene_start"].values, 2),
                "end": np.repeat(ds_cnv["gene_end"].values, 2),
                "windows": np.repeat(ds_cnv["gene_windows"].values, 2),
                # alternate amplification and deletion
                "cnv_type": np.tile(np.array(["amp", "del"]), n_genes),
                "max_af": max_af,
                "gene_id": np.repeat(ds_cnv["gene_id"].values, 2),
                "gene_name": np.repeat(ds_cnv["gene_name"].values, 2),
                "gene_strand": np.repeat(ds_cnv["gene_strand"].values, 2),
            }
        )

        # add variant label
        df_variants["label"] = df_variants.apply(_make_gene_cnv_label, axis="columns")

        # build the output dataset
        ds_out = xr.Dataset()

        # cohort variables
        for coh_col in df_cohorts.columns:
            ds_out[f"cohort_{coh_col}"] = "cohorts", df_cohorts[coh_col]

        # variant variables
        for snp_col in df_variants.columns:
            ds_out[f"variant_{snp_col}"] = "variants", df_variants[snp_col]

        # event variables
        ds_out["event_count"] = ("variants", "cohorts"), count
        ds_out["event_nobs"] = ("variants", "cohorts"), nobs
        ds_out["event_frequency"] = ("variants", "cohorts"), frequency

        # deal with invariants
        if drop_invariant:
            loc_variant = df_variants["max_af"].values > 0
            ds_out = ds_out.isel(variants=loc_variant)
            df_variants = df_variants.loc[loc_variant]

        # apply variant query
        if variant_query is not None:
            loc_variants = df_variants.eval(variant_query).values
            ds_out = ds_out.isel(variants=loc_variants)

        # add confidence intervals
        _add_frequency_ci(ds_out, ci_method)

        # tidy up display by sorting variables
        ds_out = ds_out[sorted(ds_out)]

        # add metadata
        title = "Gene CNV frequencies"
        ds_out.attrs["title"] = title

        return ds_out

    @staticmethod
    def plot_frequencies_time_series(ds, height=None, width=None, title=True, **kwargs):
        """Create a time series plot of variant frequencies using plotly.

        Parameters
        ----------
        ds : xarray.Dataset
            A dataset of variant frequencies, such as returned by `Ag3.snp_allele_frequencies_advanced()`,
            `Ag3.aa_allele_frequencies_advanced()` or `Ag3.gene_cnv_frequencies_advanced()`.
        height : int, optional
            Height of plot in pixels.
        width : int, optional
            Width of plot in pixels
        title : bool or str, optional
            If True, attempt to use metadata from input dataset as a plot
            title. Otherwise, use supplied value as a title.
        **kwargs
            Passed through to `px.line()`.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            A plotly figure containing line graphs. The resulting figure will have one
            panel per cohort, grouped into columns by taxon, and grouped into rows by
            area. Markers and lines show frequencies of variants.

        """

        # handle title
        if title is True:
            title = ds.attrs.get("title", None)

        # extract cohorts into a dataframe
        cohort_vars = [v for v in ds if v.startswith("cohort_")]
        df_cohorts = ds[cohort_vars].to_dataframe()
        df_cohorts.columns = [c.split("cohort_")[1] for c in df_cohorts.columns]

        # extract variant labels
        variant_labels = ds["variant_label"].values

        # build a long-form dataframe from the dataset
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

        # remove events with no observations
        df_events = df_events.query("nobs > 0")

        # calculate error bars
        frq = df_events["frequency"]
        frq_ci_low = df_events["frequency_ci_low"]
        frq_ci_upp = df_events["frequency_ci_upp"]
        df_events["frequency_error"] = frq_ci_upp - frq
        df_events["frequency_error_minus"] = frq - frq_ci_low

        # make a plot
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
            **kwargs,
        )

        # set Y axis limits
        fig.update_layout(yaxis_range=[-0.05, 1.05])

        return fig

    @staticmethod
    def plot_frequencies_map_markers(m, ds, variant, taxon, period, clear=True):
        """Plot markers on a map showing variant frequencies for cohorts grouped
        by area (space), period (time) and taxon.

        Parameters
        ----------
        m : ipyleaflet.Map
            The map on which to add the markers.
        ds : xarray.Dataset
            A dataset of variant frequencies, such as returned by `Ag3.snp_allele_frequencies_advanced()`,
            `Ag3.aa_allele_frequencies_advanced()` or `Ag3.gene_cnv_frequencies_advanced()`.
        variant : int or str
            Index or label of variant to plot.
        taxon : str
            Taxon to show markers for.
        period : pd.Period
            Time period to show markers for.
        clear : bool, optional
            If True, clear all layers (except the base layer) from the map before
            adding new markers.

        """

        # slice dataset to variant of interest
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

        # convert to a dataframe for convenience
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

        # select data matching taxon and period parameters
        df_markers = df_markers.loc[
            (
                (df_markers["cohort_taxon"] == taxon)
                & (df_markers["cohort_period"] == period)
            )
        ]

        # clear existing layers in the map
        if clear:
            for layer in m.layers[1:]:
                m.remove_layer(layer)

        # add markers
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
                taxon: {x.cohort_taxon} <br/>
                area: {x.cohort_area} <br/>
                period: {x.cohort_period} <br/>
                sample size: {x.cohort_size} <br/>
                frequency: {x.event_frequency:.0%}
                (95% CI: {x.event_frequency_ci_low:.0%} - {x.event_frequency_ci_upp:.0%})
            """
            marker.popup = ipyleaflet.Popup(
                child=ipywidgets.HTML(popup_html),
            )
            m.add_layer(marker)

    @staticmethod
    def plot_frequencies_interactive_map(
        ds,
        center=(-2, 20),
        zoom=3,
        title=True,
        epilogue="""
            Variant frequencies are shown as coloured markers. Opacity of color
            denotes frequency. Click on a marker for more information.
        """,
    ):
        """Create an interactive map with markers showing variant frequencies for cohorts
        grouped by area (space), period (time) and taxon.

        Parameters
        ----------
        ds : xarray.Dataset
            A dataset of variant frequencies, such as returned by `Ag3.snp_allele_frequencies_advanced()`,
            `Ag3.aa_allele_frequencies_advanced()` or `Ag3.gene_cnv_frequencies_advanced()`.
        center : tuple of int, optional
            Location to center the map.
        zoom : int, optional
            Initial zoom level.
        title : bool or str, optional
            If True, attempt to use metadata from input dataset as a plot
            title. Otherwise, use supplied value as a title.
        epilogue : str, optional
            Additional text to display below the map.

        Returns
        -------
        out : ipywidgets.Widget
            An interactive map with widgets for selecting which variant, taxon and
            time period to display.

        """

        # handle title
        if title is True:
            title = ds.attrs.get("title", None)

        # create a map
        freq_map = ipyleaflet.Map(center=center, zoom=zoom)

        # setup interactive controls
        variants = ds["variant_label"].values
        taxa = np.unique(ds["cohort_taxon"].values)
        periods = np.unique(ds["cohort_period"].values)
        controls = ipywidgets.interactive(
            Ag3.plot_frequencies_map_markers,
            m=ipywidgets.fixed(freq_map),
            ds=ipywidgets.fixed(ds),
            variant=ipywidgets.Dropdown(options=variants, description="variant: "),
            taxon=ipywidgets.Dropdown(options=taxa, description="taxon: "),
            period=ipywidgets.Dropdown(options=periods, description="period: "),
            clear=ipywidgets.fixed(True),
        )

        # lay out widgets
        components = []
        if title is not None:
            components.append(ipywidgets.HTML(value=f"<h3>{title}</h3>"))
        components.append(controls)
        components.append(freq_map)
        if epilogue is not None:
            components.append(ipywidgets.HTML(value=f"{epilogue}"))

        out = ipywidgets.VBox(components)

        return out


@numba.njit("Tuple((int8, int64))(int8[:], int8)")
def _cn_mode_1d(a, vmax):

    # setup intermediates
    m = a.shape[0]
    counts = np.zeros(vmax + 1, dtype=numba.int64)

    # initialise return values
    mode = numba.int8(-1)
    mode_count = numba.int64(0)

    # iterate over array values, keeping track of counts
    for i in range(m):
        v = a[i]
        if 0 <= v <= vmax:
            c = counts[v]
            c += 1
            counts[v] = c
            if c > mode_count:
                mode = v
                mode_count = c
            elif c == mode_count and v < mode:
                # consistency with scipy.stats, break ties by taking lower value
                mode = v

    return mode, mode_count


@numba.njit("Tuple((int8[:], int64[:]))(int8[:, :], int8)")
def _cn_mode(a, vmax):

    # setup intermediates
    n = a.shape[1]

    # setup outputs
    modes = np.zeros(n, dtype=numba.int8)
    counts = np.zeros(n, dtype=numba.int64)

    # iterate over columns, computing modes
    for j in range(a.shape[1]):
        mode, count = _cn_mode_1d(a[:, j], vmax)
        modes[j] = mode
        counts[j] = count

    return modes, counts


def _make_sample_period_month(row):
    year = row.year
    month = row.month
    if year > 0 and month > 0:
        return pd.Period(freq="M", year=year, month=month)
    else:
        return pd.NaT


def _make_sample_period_quarter(row):
    year = row.year
    month = row.month
    if year > 0 and month > 0:
        return pd.Period(freq="Q", year=year, month=month)
    else:
        return pd.NaT


def _make_sample_period_year(row):
    year = row.year
    if year > 0:
        return pd.Period(freq="A", year=year)
    else:
        return pd.NaT


# def _genotypes_to_alt_allele_counts_melt(gt, max_allele):
#     """Convert a genotype array to an array of alt allele counts, melted to
#     store one row per alt allele."""
#
#     n_variants = gt.shape[0]
#     n_samples = gt.shape[1]
#
#     # convert to genotype allele counts
#     gac = allel.GenotypeArray(gt).to_allele_counts(max_allele=max_allele)
#     assert gac.shape == (n_variants, n_samples, max_allele + 1)
#
#     # sum total observations over alleles
#     gan = gac.sum(axis=2)
#
#     # keep only alt allele counts
#     gac_alt = gac[:, :, 1:]
#     assert gac_alt.shape == (n_variants, n_samples, max_allele)
#
#     # use some numpy tricks to melt alleles into rows
#     gac_alt_melt = gac_alt.swapaxes(2, 1).reshape(-1, n_samples)
#     assert gac_alt_melt.shape == (n_variants * max_allele, n_samples)
#     gan_melt = np.repeat(gan, max_allele, axis=0)
#     assert gan_melt.shape == (n_variants * max_allele, n_samples)
#
#     return gac_alt_melt, gan_melt


@numba.njit
def _genotypes_to_alt_allele_counts_melt_kernel(gt, max_allele):
    """Convert a genotype array to an array of alt allele counts, melted to
    store one row per alt allele."""

    n_variants = gt.shape[0]
    n_samples = gt.shape[1]
    ploidy = gt.shape[2]

    gac_alt_melt = np.zeros((n_variants * max_allele, n_samples), dtype=np.uint8)
    gan = np.zeros((n_variants, n_samples), dtype=np.uint8)

    for i in range(n_variants):
        out_i_offset = (i * max_allele) - 1
        for j in range(n_samples):
            for k in range(ploidy):
                allele = gt[i, j, k]
                if allele > 0:
                    out_i = out_i_offset + allele
                    gac_alt_melt[out_i, j] += 1
                    gan[i, j] += 1
                elif allele == 0:
                    gan[i, j] += 1

    return gac_alt_melt, gan


def _genotypes_to_alt_allele_counts_melt(gt, max_allele):
    gac_alt_melt, gan = _genotypes_to_alt_allele_counts_melt_kernel(gt, max_allele)
    gan_melt = np.repeat(gan, max_allele, axis=0)
    return gac_alt_melt, gan_melt


@numba.njit
def _take_sum_cols(a, indices):
    n_variants = a.shape[0]
    n_indices = indices.shape[0]
    out = np.zeros(n_variants, dtype=np.int64)
    for i in range(n_variants):
        v_sum = 0
        for j in range(n_indices):
            ix = indices[j]
            v = a[i, ix]
            v_sum += v
        out[i] = v_sum
    return out


def _make_snp_label(row):
    label = (
        f"{row['contig']}:{row['position']:,} {row['ref_allele']}>{row['alt_allele']}"
    )
    return label


def _make_snp_label_effect(row):
    label = (
        f"{row['contig']}:{row['position']:,} {row['ref_allele']}>{row['alt_allele']}"
    )
    aa_change = row["aa_change"]
    if isinstance(aa_change, str):
        label += f" ({aa_change})"
    return label


def _make_snp_label_aa(row):
    label = f"{row['aa_change']} ({row['contig']}:{row['position']:,} {row['ref_allele']}>{row['alt_allele']})"
    return label


def _make_gene_cnv_label(row):
    label = row["gene_id"]
    gene_name = row["gene_name"]
    if isinstance(gene_name, str):
        label += f" ({gene_name})"
    label += f" {row['cnv_type']}"
    return label


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
        ds_out["variant_alt_allele"] = "variants", np.array([alt_allele], dtype=object)

    return ds_out


def _add_frequency_ci(ds, ci_method):
    if ci_method is not None:
        count = ds["event_count"].values
        nobs = ds["event_nobs"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            frq_ci_low, frq_ci_upp = proportion_confint(
                count=count, nobs=nobs, method=ci_method
            )
        ds["event_frequency_ci_low"] = ("variants", "cohorts"), frq_ci_low
        ds["event_frequency_ci_upp"] = ("variants", "cohorts"), frq_ci_upp


def _prep_samples_for_cohort_grouping(*, df_samples, area_by, period_by):

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
        make_period = _make_sample_period_year
    elif period_by == "quarter":
        make_period = _make_sample_period_quarter
    elif period_by == "month":
        make_period = _make_sample_period_month
    else:
        raise ValueError(
            f"Value for period_by parameter must be one of 'year', 'quarter', 'month'; found {period_by!r}."
        )
    sample_period = df_samples.apply(make_period, axis="columns")
    df_samples["period"] = sample_period

    # add area column for consistent output
    df_samples["area"] = df_samples[area_by]

    return df_samples


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
    df_cohorts = df_cohorts.query(f"size >= {min_cohort_size}").reset_index(drop=True)

    return df_cohorts


def _check_param_min_cohort_size(min_cohort_size):
    if not isinstance(min_cohort_size, int):
        raise TypeError(
            f"Type of parameter min_cohort_size must be int; found {type(min_cohort_size)}."
        )
    if min_cohort_size < 1:
        raise ValueError(
            f"Value of parameter min_cohort_size must be at least 1; found {min_cohort_size}."
        )
