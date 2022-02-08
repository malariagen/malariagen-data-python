import re
from bisect import bisect_left, bisect_right
from collections import namedtuple

import allel
import dask.array as da
import numba
import numpy as np
import pandas
import xarray
import zarr

from . import veff
from .util import (
    DIM_ALLELE,
    DIM_PLOIDY,
    DIM_SAMPLE,
    DIM_VARIANT,
    da_compress,
    da_from_zarr,
    dask_compress_dataset,
    init_filesystem,
    init_zarr_store,
    read_gff3,
    unpack_gff3_attributes,
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

Region = namedtuple("Region", ["contig", "start", "end"])


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
    """Compatibility function, allows us to use release identifiers like "3.0" and "3.1"
    in the public API, and map these internally into storage path segments."""
    if release == "3.0":
        # special case
        return "v3"
    elif release.startswith("3."):
        return f"v{release}"
    else:
        raise ValueError(f"Invalid release: {release!r}")


def _path_to_release(path):
    """Compatibility function, allows us to use release identifiers like "3.0" and "3.1"
    in the public API, and map these internally into storage path segments."""
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
        Base path to data. Give "gs://vo_agam_release/" to use Google Cloud Storage,
        or a local path on your file system if data have been downloaded.
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
            f"Releases available    : {','.join(self.releases)}\n"
            f"Cohorts analysis      : {DEFAULT_COHORTS_ANALYSIS}\n"
            f"Species analysis      : {DEFAULT_SPECIES_ANALYSIS}\n"
            f"Site filters analysis : {DEFAULT_SITE_FILTERS_ANALYSIS}\n"
            f"---\n"
            f"Please note that data are subject to terms of use,\n"
            f"for more information see https://www.malariagen.net/data\n"
            f"or contact data@malariagen.net.\n"
            f"---\n"
            f"For API documentation see https://malariagen.github.io/vector-data/ag3/api.html"
        )

    def _repr_html_(self):
        return f"""
            <style type="text/css">
                table.malariagen-ag3 th, table.malariagen-ag3 td {{
                    text-align: left
                }}
            </style>
            <table class="malariagen-ag3">
                <thead>
                    <tr>
                        <th colspan=2>MalariaGEN Ag3 data resource API</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><th>Storage URL</th><td>{self._url}</td></tr>
                    <tr><th>Releases available</th><td>{','.join(self.releases)}</td></tr>
                    <tr><th>Cohorts analysis</th><td>{DEFAULT_COHORTS_ANALYSIS}</td></tr>
                    <tr><th>Species analysis</th><td>{DEFAULT_SPECIES_ANALYSIS}</td></tr>
                    <tr><th>Site filters analysis</th><td>{DEFAULT_SITE_FILTERS_ANALYSIS}</td></tr>
                </tbody>
            </table>
            <p>Please note that data are subject to terms of use,
            for more information see <a href="https://www.malariagen.net/data">
            the MalariaGEN website</a> or contact data@malariagen.net.</p>
            <p>See also the <a href="https://malariagen.github.io/vector-data/ag3/api.html">Ag3 API docs</a>.</p>
        """

    @property
    def releases(self):
        """The releases for which data are available at the given storage location."""
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
            df = pandas.read_csv(f, sep="\t", na_values="")
        df["release"] = release
        return df

    def sample_sets(self, release=None):
        """Access a dataframe of sample sets.

        Parameters
        ----------
        release : str, optional
            Release identifier. Give "3.0" to access the Ag1000G phase 3 data release.

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
            # retrieve sample sets from multiple releases
            df = pandas.concat(
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
                df = pandas.read_csv(f, na_values="")

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
            if analysis == "aim_20200422":
                path = f"{self._base_path}/{release_path}/metadata/species_calls_20200422/{sample_set}/samples.species_aim.csv"
            elif analysis == "pca_20200422":
                path = f"{self._base_path}/{release_path}/metadata/species_calls_20200422/{sample_set}/samples.species_pca.csv"
            else:
                raise ValueError(f"Unknown species calling analysis: {analysis!r}")
            with self._fs.open(path) as f:
                df = pandas.read_csv(
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

            self._cache_species_calls[key] = df
            return df

    def _prep_sample_sets_arg(self, *, sample_sets):
        """Common handling for the `sample_sets` parameter. For convenience, we allow this
        to be a single sample set, or a list of sample sets, or a release identifier, or a
        list of release identifiers."""

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

        return sample_sets

    def _resolve_region(self, region):
        """Parse the provided region and return `Region(contig, start, end)`.
        Supports contig names, gene names and genomic coordinates"""

        # region is already Region tuple
        if isinstance(region, Region):
            return region

        # check type, fail early if bad
        if not isinstance(region, str):
            raise TypeError("The region parameter must be a string or Region object.")

        # check if region is a chromosome arm
        if region in self.contigs:
            return Region(region, None, None)

        # check if region is a region string
        region_pattern_match = re.search(r"([a-zA-Z0-9]+)\:(.+)\-(.+)", region)
        if region_pattern_match:
            # parse region string that contains genomic coordinates
            region_split = region_pattern_match.groups()

            contig = region_split[0]
            start = int(region_split[1].replace(",", ""))
            end = int(region_split[2].replace(",", ""))

            if contig not in self.contigs:
                raise ValueError(f"Contig {contig} does not exist in the dataset.")
            elif (
                start < 0
                or end <= start
                or end > self.genome_sequence(region=contig).shape[0]
            ):
                raise ValueError("Provided genomic coordinates are not valid.")

            return Region(contig, start, end)

        # check if region is a gene annotation feature ID
        gene_annotation = self.geneset(["ID"]).query(f"ID == '{region}'")
        if not gene_annotation.empty:
            # region is a feature ID
            gene_annotation = gene_annotation.squeeze()
            return Region(
                gene_annotation.contig, gene_annotation.start, gene_annotation.end
            )

        raise ValueError(
            f"Region {region!r} is not a valid contig, region string or feature ID."
        )

    def locate_region(self, region):
        """Get array slice and a parsed genomic region.

        Parameters
        ----------
        region : str or Region
            Can be a string with chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"),
            genomic region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.

        Returns
        -------
        loc_region, region : slice, Region
        """
        region = self._resolve_region(region)
        root = self.open_snp_sites()
        pos = allel.SortedIndex(root[region.contig]["variants"]["POS"])
        loc_region = pos.locate_range(region.start, region.end)

        return loc_region, region

    def species_calls(self, sample_sets=None, analysis=DEFAULT_SPECIES_ANALYSIS):
        """Access species calls for one or more sample sets.

        Parameters
        ----------
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"] or a release identifier (e.g.,
            "3.0") or a list of release identifiers.
        analysis : {"aim_20200422", "pca_20200422"}
            Species calling analysis.

        Returns
        -------
        df : pandas.DataFrame

        """

        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)

        # concatenate multiple sample sets
        dfs = [
            self._read_species_calls(sample_set=s, analysis=analysis)
            for s in sample_sets
        ]
        df = pandas.concat(dfs, axis=0, ignore_index=True)

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
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "3.0") or a list of release identifiers.
        species_analysis : {"aim_20200422", "pca_20200422"}, optional
            Include species calls in metadata.
        cohorts_analysis : str
            Cohort analysis identifier (date of analysis), optional,  default is latest version.
            Includes sample cohort calls in metadata.

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
        df = pandas.concat(dfs, axis=0, ignore_index=True)

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
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic region
            defined with coordinates (e.g., "2L:44989425-44998059") or a named tuple with
            genomic location `Region(contig, start, end)`. Multiple values can be provided
            as a list, in which case data will be concatenated, e.g., ["3R", "AGAP005958"].
        mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Mask to use.
        field : str, optional
            Array to access.
        analysis : str, optional
            Site filters analysis version.
        inline_array : bool, optional
            Passed through to dask.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr chunks.
            Also can be a target size, e.g., '200 MiB'.

        Returns
        -------
        d : dask.array.Array

        """

        if isinstance(region, (list, tuple)) and not isinstance(region, Region):
            return da.concatenate(
                [
                    self.site_filters(
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
        else:
            loc_region, region = self.locate_region(region)
            root = self.open_site_filters(mask=mask, analysis=analysis)
            z = root[region.contig]["variants"][field]
            d = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
            return d[loc_region]

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

    def snp_sites(
        self,
        region,
        field=None,
        site_mask=None,
        site_filters_analysis=DEFAULT_SITE_FILTERS_ANALYSIS,
        inline_array=True,
        chunks="native",
    ):
        """Access SNP site data (positions and alleles).

        Parameters
        ----------
        region: str or list of str or Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic region
            defined with coordinates (e.g., "2L:44989425-44998059") or a named tuple with
            genomic location `Region(contig, start, end)`. Multiple values can be provided
            as a list, in which case data will be concatenated, e.g., ["3R", "AGAP005958"].
        field : {"POS", "REF", "ALT"}, optional
            Array to access. If not provided, all three arrays POS, REF, ALT will be returned as a
            tuple.
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
        d : dask.array.Array or tuple of dask.array.Array

        """

        if field is None:
            # return POS, REF, ALT
            ret = tuple(
                self.snp_sites(
                    region=region,
                    field=f,
                    site_mask=None,
                    chunks=chunks,
                    inline_array=inline_array,
                )
                for f in ("POS", "REF", "ALT")
            )

        elif isinstance(region, (tuple, list)) and not isinstance(region, Region):
            # concatenate
            ret = da.concatenate(
                [
                    self.snp_sites(
                        region=r,
                        field=field,
                        site_mask=None,
                        chunks=chunks,
                        inline_array=inline_array,
                    )
                    for r in region
                ],
                axis=0,
            )

        else:
            loc_region, region = self.locate_region(region)
            root = self.open_snp_sites()
            z = root[region.contig]["variants"][field]
            ret = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
            ret = ret[loc_region]

        if site_mask is not None:
            loc_sites = self.site_filters(
                region=region,
                mask=site_mask,
                analysis=site_filters_analysis,
                chunks=chunks,
                inline_array=inline_array,
            )
            if isinstance(ret, tuple):
                ret = tuple(da_compress(loc_sites, d, axis=0) for d in ret)
            else:
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
        loc_region, region = self.locate_region(region)
        root = self.open_snp_genotypes(sample_set=sample_set)
        z = root[region.contig]["calldata"][field]

        d = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        return d[loc_region]

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
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic region
            defined with coordinates (e.g., "2L:44989425-44998059") or a named tuple with
            genomic location `Region(contig, start, end)`. Multiple values can be provided
            as a list, in which case data will be concatenated, e.g., ["3R", "AGAP005958"].
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "3.0") or a list of release identifiers.
        field : {"GT", "GQ", "AD", "MQ"}
            Array to access.
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_filters_analysis : str, optional
            Site filters analysis version.
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr chunks.
            Also can be a target size, e.g., '200 MiB'.

        Returns
        -------
        d : dask.array.Array

        """

        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)

        # normalise to simplify concatenation logic
        if isinstance(region, str) or isinstance(region, Region):
            region = [region]

        # concatenate multiple sample sets and/or contigs
        d = da.concatenate(
            [
                da.concatenate(
                    [
                        self._snp_genotypes(
                            region=r,
                            sample_set=s,
                            field=field,
                            inline_array=inline_array,
                            chunks=chunks,
                        )
                        for s in sample_sets
                    ],
                    axis=1,
                )
                for r in region
            ],
            axis=0,
        )

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
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic region
            defined with coordinates (e.g., "2L:44989425-44998059") or a named tuple with
            genomic location `Region(contig, start, end)`.
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr chunks.
            Also can be a target size, e.g., '200 MiB'.

        Returns
        -------
        d : dask.array.Array

        """
        genome = self.open_genome()
        region = self._resolve_region(region)
        z = genome[region.contig]
        d = da_from_zarr(z, inline_array=inline_array, chunks=chunks)

        if region.start and region.end:
            loc_region = slice(region.start - 1, region.end)
        else:
            loc_region = slice(None, None)

        return d[loc_region]

    def geneset(self, attributes=("ID", "Parent", "Name", "description")):
        """Access genome feature annotations (AgamP4.12).

        Parameters
        ----------
        attributes : list of str, optional
            Attribute keys to unpack into columns. Provide "*" to unpack all attributes.

        Returns
        -------
        df : pandas.DataFrame

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

    def is_accessible(
        self, region, site_mask, site_filters_analysis=DEFAULT_SITE_FILTERS_ANALYSIS
    ):
        """Compute genome accessibility array.

        Parameters
        ----------
        region: str or list of str or Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic region
            defined with coordinates (e.g., "2L:44989425-44998059") or a named tuple with
            genomic location `Region(contig, start, end)`.
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_filters_analysis : str, optional
            Site filters analysis version.

        Returns
        -------
        a : numpy.ndarray

        """

        # resolve region
        region = self._resolve_region(region=region)

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

    def _site_mask_ids(self, *, site_filters_analysis):
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

        loc_feature, _ = self.locate_region(region)

        # grab pos, ref and alt for chrom arm from snp_sites
        pos, ref, alt = self.snp_sites(region=region)
        ref = ref.compute()
        alt = alt.compute()

        # access site filters
        filter_pass = dict()
        masks = self._site_mask_ids(site_filters_analysis=site_filters_analysis)
        for m in masks:
            x = self.site_filters(region=region, mask=m, analysis=site_filters_analysis)
            x = x.compute()
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
        df_snps = pandas.DataFrame(cols)

        return region, loc_feature, df_snps

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

        """

        # setup initial dataframe of SNPs
        _, _, df_snps = self._snp_df(
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
            # remove the nan rows
            for coh in df_coh[cohorts].unique():
                if isinstance(coh, str):
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
            Gene transcript ID (AgamP4.12), e.g., "AGAP004707-RA".
        cohorts : str or dict
            If a string, gives the name of a predefined cohort set, e.g., one of
            {"admin1_month", "admin1_year", "admin2_month", "admin2_year"}.
            If a dict, should map cohort labels to sample queries, e.g.,
            `{"bf_2012_col": "country == 'Burkina Faso' and year == 2012 and species == 'coluzzii'"}`.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample metadata e.g.,
            "species == 'coluzzii' and country == 'Burkina Faso'".
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
        effects : bool, optional
            If True, add SNP effect columns.

        Returns
        -------
        df_snps : pandas.DataFrame

        Notes
        -----
        Cohorts with fewer samples than min_cohort_size will be excluded from output.

        """

        # handle sample_query
        loc_samples = None
        if sample_query is not None:
            df_samples = self.sample_metadata(
                sample_sets=sample_sets,
                cohorts_analysis=cohorts_analysis,
                species_analysis=species_analysis,
            )
            loc_samples = df_samples.eval(sample_query).values

        # setup initial dataframe of SNPs
        region, loc_feature, df_snps = self._snp_df(
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
        df_freqs = pandas.DataFrame(freq_cols)

        # build the final dataframe
        df_snps.reset_index(drop=True, inplace=True)
        df_snps = pandas.concat([df_snps, df_freqs], axis=1)

        # add max allele freq column (concat here also reduces fragmentation)
        df_snps = pandas.concat(
            [
                df_snps,
                pandas.DataFrame(
                    {"max_af": df_snps[list(freq_cols.keys())].max(axis=1)}
                ),
            ],
            axis=1,
        )

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
            ann = self._annotator()
            ann.get_effects(transcript=transcript, variants=df_snps)

        return df_snps

    def cross_metadata(self):
        """Load a dataframe containing metadata about samples in colony crosses, including
        which samples are parents or progeny in which crosses.

        Returns
        -------
        df : pandas.DataFrame

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
                df = pandas.read_csv(
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

        """

        # access the array of values for all genome positions
        root = self.open_site_annotations()

        # resolve region
        region = self._resolve_region(region)

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
        self, *, region, sample_set, site_filters_analysis, inline_array, chunks
    ):

        region = self._resolve_region(region)

        coords = dict()
        data_vars = dict()

        # variant arrays
        sites_root = self.open_snp_sites()

        # variant_position
        pos_z = sites_root[f"{region.contig}/variants/POS"]

        loc_region, region = self.locate_region(region)
        variant_position = da_from_zarr(pos_z, inline_array=inline_array, chunks=chunks)
        coords["variant_position"] = [DIM_VARIANT], variant_position[loc_region]

        # variant_allele
        ref_z = sites_root[f"{region.contig}/variants/REF"]
        alt_z = sites_root[f"{region.contig}/variants/ALT"]
        ref = da_from_zarr(ref_z, inline_array=inline_array, chunks=chunks)
        alt = da_from_zarr(alt_z, inline_array=inline_array, chunks=chunks)
        variant_allele = da.concatenate(
            [ref[loc_region, None], alt[loc_region]], axis=1
        )
        data_vars["variant_allele"] = [DIM_VARIANT, DIM_ALLELE], variant_allele

        # variant_contig
        contig_index = self.contigs.index(region.contig)
        variant_contig = da.full_like(
            variant_position, fill_value=contig_index, dtype="u1"
        )
        coords["variant_contig"] = [DIM_VARIANT], variant_contig[loc_region]

        # site filters arrays
        for mask in "gamb_colu_arab", "gamb_colu", "arab":
            filters_root = self.open_site_filters(
                mask=mask, analysis=site_filters_analysis
            )
            z = filters_root[f"{region.contig}/variants/filter_pass"]
            d = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
            data_vars[f"variant_filter_pass_{mask}"] = [DIM_VARIANT], d[loc_region]

        # call arrays
        calls_root = self.open_snp_genotypes(sample_set=sample_set)
        gt_z = calls_root[f"{region.contig}/calldata/GT"]
        call_genotype = da_from_zarr(gt_z, inline_array=inline_array, chunks=chunks)
        gq_z = calls_root[f"{region.contig}/calldata/GQ"]
        call_gq = da_from_zarr(gq_z, inline_array=inline_array, chunks=chunks)
        ad_z = calls_root[f"{region.contig}/calldata/AD"]
        call_ad = da_from_zarr(ad_z, inline_array=inline_array, chunks=chunks)
        mq_z = calls_root[f"{region.contig}/calldata/MQ"]
        call_mq = da_from_zarr(mq_z, inline_array=inline_array, chunks=chunks)
        data_vars["call_genotype"] = (
            [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY],
            call_genotype[loc_region],
        )
        data_vars["call_GQ"] = ([DIM_VARIANT, DIM_SAMPLE], call_gq[loc_region])
        data_vars["call_MQ"] = ([DIM_VARIANT, DIM_SAMPLE], call_mq[loc_region])
        data_vars["call_AD"] = (
            [DIM_VARIANT, DIM_SAMPLE, DIM_ALLELE],
            call_ad[loc_region],
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
        ds = xarray.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

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

        """

        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)

        # normalise to simplify concatenation logic
        if isinstance(region, str) or isinstance(region, Region):
            region = [region]

        # concatenate multiple sample sets and/or contigs
        ds = xarray.concat(
            [
                xarray.concat(
                    [
                        self._snp_calls_dataset(
                            region=r,
                            sample_set=s,
                            site_filters_analysis=site_filters_analysis,
                            inline_array=inline_array,
                            chunks=chunks,
                        )
                        for s in sample_sets
                    ],
                    dim=DIM_SAMPLE,
                    data_vars="minimal",
                    coords="minimal",
                    compat="override",
                    join="override",
                )
                for r in region
            ],
            dim=DIM_VARIANT,
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="override",
        )

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
        ds = xarray.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        return ds

    def cnv_hmm(
        self,
        contig,
        sample_sets=None,
        inline_array=True,
        chunks="native",
    ):
        """Access CNV HMM data.

        Parameters
        ----------
        contig : str
            Chromosome arm, e.g., "3R".
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

        """

        # TODO support multiple contigs?

        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)

        # concatenate sample sets along samples dimension
        datasets = [
            self._cnv_hmm_dataset(
                contig=contig,
                sample_set=s,
                inline_array=inline_array,
                chunks=chunks,
            )
            for s in sample_sets
        ]
        ds = xarray.concat(
            datasets,
            dim=DIM_SAMPLE,
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="override",
        )

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

    def cnv_coverage_calls(
        self,
        contig,
        sample_set,
        analysis,
        inline_array=True,
        chunks="native",
    ):
        """Access CNV HMM data.

        Parameters
        ----------
        contig : str
            Chromosome arm, e.g., "3R".
        sample_set : str
            Sample set identifier.
        analysis : {'gamb_colu', 'arab', 'crosses'}
            Name of CNV analysis.
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr chunks.
            Also can be a target size, e.g., '200 MiB'.

        Returns
        -------
        ds : xarray.Dataset

        """

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
        ds = xarray.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

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
        ds = xarray.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

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
            Chromosome arm, e.g., "3R".
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

        """

        # TODO support multiple contigs

        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)

        # concatenate sample sets
        datasets = [
            self._cnv_discordant_read_calls_dataset(
                contig=contig,
                sample_set=s,
                inline_array=inline_array,
                chunks=chunks,
            )
            for s in sample_sets
        ]
        ds = xarray.concat(
            datasets,
            dim=DIM_SAMPLE,
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="override",
        )

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
        ds_out = xarray.Dataset(
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
            `{"bf_2012_col": "country == 'Burkina Faso' and year == 2012 and species == 'coluzzii'"}`.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample metadata e.g.,
            "species == 'coluzzii' and country == 'Burkina Faso'".
        cohorts_analysis : str
            Cohort analysis identifier (date of analysis), default is latest version.
        min_cohort_size : int
            Minimum cohort size, below which allele frequencies are not calculated for cohorts.
            Please note, NaNs will be returned for any cohorts with fewer samples than min_cohort_size,
            these can be removed from the output dataframe using pandas df.dropna(axis='columns').
        species_analysis : {"aim_20200422", "pca_20200422"}, optional
            Include species calls in metadata.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "3.0") or a list of release identifiers.

        Returns
        -------
        df : pandas.DataFrame

        Notes
        -----
        NaNs will be returned for any cohorts with fewer samples than min_cohort_size,
        these can be removed from the output dataframe using pandas df.dropna(axis='columns').

        """

        # handle sample_query
        loc_samples = None
        if sample_query is not None:
            df_samples = self.sample_metadata(
                sample_sets=sample_sets,
                cohorts_analysis=cohorts_analysis,
                species_analysis=species_analysis,
            )
            loc_samples = df_samples.eval(sample_query).values

        # get gene copy number data
        ds_cnv = self.gene_cnv(contig=contig, sample_sets=sample_sets)

        # get sample metadata
        df_meta = self.sample_metadata(sample_sets=sample_sets)

        # get genes
        df_genes = self.geneset().query(f"type == 'gene' and contig == '{contig}'")

        # figure out expected copy number
        if contig == "X":
            is_male = (df_meta["sex_call"] == "M").values
            expected_cn = np.where(is_male, 1, 2)[np.newaxis, :]
        else:
            expected_cn = 2

        # setup output dataframe
        df = df_genes.copy()
        # drop columns we don't need
        df.drop(columns=["source", "type", "score", "phase", "Parent"], inplace=True)

        # setup intermediates
        cn = ds_cnv["CN_mode"].values
        is_amp = cn > expected_cn
        is_del = (cn >= 0) & (cn < expected_cn)

        # set up cohort dict
        # build coh dict
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
                freq_cols[f"frq_{coh}_amp"] = amp_freq_coh
                freq_cols[f"frq_{coh}_del"] = del_freq_coh

        # build a dataframe with the frequency columns
        df_freqs = pandas.DataFrame(freq_cols)

        # build the final dataframe
        df.reset_index(drop=True, inplace=True)
        df = pandas.concat([df, df_freqs], axis=1)

        # set gene ID as index for convenience
        df.set_index("ID", inplace=True)

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
            best. Otherwise use the "gamb_colu_arab" analysis.

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
            if ".zmetadata" not in store:
                root = None
            else:
                root = zarr.open_consolidated(store=store)
            self._cache_haplotypes[(sample_set, analysis)] = root
        return root

    def open_haplotype_sites(self, analysis):
        """Open haplotype sites zarr.

        Parameters
        ----------
        analysis : {"arab", "gamb_colu", "gamb_colu_arab"}
            Which phasing analysis to use. If analysing only An. arabiensis, the "arab" analysis
            is best. If analysing only An. gambiae and An. coluzzii, the "gamb_colu" analysis is
            best. Otherwise use the "gamb_colu_arab" analysis.

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
        ds = xarray.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        return ds

    def haplotypes(
        self,
        contig,
        analysis,
        sample_sets=None,
        inline_array=True,
        chunks="native",
    ):
        """Access haplotype data.

        Parameters
        ----------
        contig : str or list of str
            Chromosome arm, e.g., "3R". Multiple values can be provided as a list,
            in which case data will be concatenated, e.g., ["3R", "3L"].
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

        """

        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)

        # normalise to simplify concatenation logic
        if isinstance(contig, str):
            contig = [contig]

        # concatenate - this is a bit gnarly, could do with simplification
        datasets = [
            [
                self._haplotypes_dataset(
                    contig=c,
                    sample_set=s,
                    analysis=analysis,
                    inline_array=inline_array,
                    chunks=chunks,
                )
                for s in sample_sets
            ]
            for c in contig
        ]
        # some sample sets have no data for a given analysis, handle this
        datasets = [[d for d in row if d is not None] for row in datasets]
        if len(datasets[0]) == 0:
            ds = None
        else:
            ds = xarray.concat(
                [
                    xarray.concat(
                        row,
                        dim=DIM_SAMPLE,
                        data_vars="minimal",
                        coords="minimal",
                        compat="override",
                        join="override",
                    )
                    for row in datasets
                ],
                dim=DIM_VARIANT,
                data_vars="minimal",
                coords="minimal",
                compat="override",
                join="override",
            )

        return ds

    def _read_cohort_metadata(self, *, sample_set, cohorts_analysis):
        """Read cohort metadata for a single sample set."""
        try:
            return self._cache_cohort_metadata[(sample_set, cohorts_analysis)]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            release_path = _release_to_path(release)
            path = f"{self._base_path}/{release_path}/metadata/cohorts_{cohorts_analysis}/{sample_set}/samples.cohorts.csv"
            with self._fs.open(path) as f:
                df = pandas.read_csv(f, na_values="")

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

        """
        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)

        # concatenate multiple sample sets
        dfs = [
            self._read_cohort_metadata(sample_set=s, cohorts_analysis=cohorts_analysis)
            for s in sample_sets
        ]
        df = pandas.concat(dfs, axis=0, ignore_index=True)

        return df

    def aa_allele_frequencies(
        self,
        transcript,
        cohorts,
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
            `{"bf_2012_col": "country == 'Burkina Faso' and year == 2012 and species == 'coluzzii'"}`.
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
        df_snps : pandas.DataFrame

        Notes
        -----
        Cohorts with fewer samples than min_cohort_size will be excluded from output.

        """

        df_snps = self.snp_allele_frequencies(
            transcript=transcript,
            cohorts=cohorts,
            cohorts_analysis=cohorts_analysis,
            min_cohort_size=min_cohort_size,
            site_mask=site_mask,
            site_filters_analysis=site_filters_analysis,
            species_analysis=species_analysis,
            sample_sets=sample_sets,
            drop_invariant=drop_invariant,
            effects=True,
        )

        # we just want aa change
        df_ns_snps = df_snps.query(
            "effect in ['NON_SYNONYMOUS_CODING', 'START_LOST', 'STOP_LOST', 'STOP_GAINED']"
        ).copy()

        # group and sum to collapse multi variant allele changes
        df_aaf = df_ns_snps.groupby(["aa_pos", "aa_change"]).sum().reset_index()
        df_aaf.set_index("aa_change", inplace=True)

        # remove old max_af
        df_aaf = df_aaf.drop(
            [
                "max_af",
                "aa_pos",
                "position",
                "pass_gamb_colu_arab",
                "pass_gamb_colu",
                "pass_arab",
            ],
            axis=1,
        ).copy()

        freq_cols = [col for col in df_aaf if col.startswith("frq")]

        # new max_af
        df_aaf["max_af"] = df_aaf[freq_cols].max(axis=1)

        return df_aaf


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
