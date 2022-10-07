import sys
from collections import Counter

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from tqdm.auto import tqdm

try:
    # noinspection PyPackageRequirements
    from google import colab
except ImportError:
    colab = None

from .util import resolve_region  # FIXME: potential confusion with self.resolve_region
from .util import (
    DIM_ALLELE,
    DIM_PLOIDY,
    DIM_SAMPLE,
    DIM_VARIANT,
    LoggingHelper,
    Region,
    da_compress,
    da_from_zarr,
    dask_compress_dataset,
    init_filesystem,
    init_zarr_store,
    locate_region,
    read_gff3,
    unpack_gff3_attributes,
    xarray_concat,
)

PUBLIC_RELEASES = ("1.0",)
GCS_URL = "gs://vo_afun_release/"
GENESET_GFF3_PATH = "reference/genome/idAnoFuneDA-416_04/GCF_943734845.2_idAnoFuneDA-416_04_genomic.gff3.gz"
GENOME_FASTA_PATH = (
    "reference/genome/idAnoFuneDA-416_04/idAnoFuneDA-416_04_1.curated_primary.fa"
)
GENOME_FAI_PATH = (
    "reference/genome/idAnoFuneDA-416_04/idAnoFuneDA-416_04_1.curated_primary.fa.fai"
)
GENOME_ZARR_PATH = (
    "reference/genome/idAnoFuneDA-416_04/idAnoFuneDA-416_04_1.curated_primary.zarr"
)

CONTIGS = "2RL", "3RL", "X"
DEFAULT_SITE_FILTERS_ANALYSIS = "dt_20200416"


def _release_to_path(release):
    """Compatibility function, allows us to use release identifiers like "1.0"
    and "1.1" in the public API, and map these internally into storage path
    segments."""
    if release == "1.0":
        # special case
        return "v1.0"  # Af1.0 does not use "v1"
    elif release.startswith("1."):
        return f"v{release}"
    else:
        raise ValueError(f"Invalid release: {release!r}")


def _path_to_release(path):
    """Compatibility function, allows us to use release identifiers like "1.0"
    and "1.1" in the public API, and map these internally into storage path
    segments."""
    if path == "v1.0":  # Af1.0 does not use "v1"
        return "1.0"
    elif path.startswith("v1."):
        return path[1:]
    else:
        raise RuntimeError(f"Unexpected release path: {path!r}")


class Af1:

    # FIXME: doc string

    contigs = CONTIGS

    def __init__(
        self,
        url=GCS_URL,
        debug=False,
        show_progress=True,
        site_filters_analysis=DEFAULT_SITE_FILTERS_ANALYSIS,
        log=sys.stdout,
        **kwargs,
    ):
        self._url = url
        self._pre = kwargs.pop("pre", False)
        self._debug = debug
        self._show_progress = show_progress
        self._site_filters_analysis = site_filters_analysis

        # set up logging
        self._log = LoggingHelper(name=__name__, out=log, debug=debug)

        # set up filesystem
        self._fs, self._base_path = init_filesystem(url, **kwargs)

        # set up caches
        self._cache_releases = None
        self._cache_sample_sets = dict()
        self._cache_sample_metadata = dict()
        self._cache_general_metadata = dict()
        self._cache_sample_set_to_release = None
        self._cache_site_filters = dict()
        self._cache_genome = None
        self._cache_snp_sites = None
        self._cache_snp_genotypes = dict()
        self._cache_geneset = dict()

    def _progress(self, iterable, **kwargs):
        # progress doesn't mix well with debug logging
        disable = self._debug or not self._show_progress
        return tqdm(iterable, disable=disable, **kwargs)

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
                            if d.startswith("v1.0")
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
            Release identifier. Give "1.0" to access the Af1.0 data release.

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
        return df.copy()

    def _prep_sample_sets_arg(self, *, sample_sets):
        """Common handling for the `sample_sets` parameter. For convenience, we
        allow this to be a single sample set, or a list of sample sets, or a
        release identifier, or a list of release identifiers."""

        if sample_sets is None:
            # all available sample sets
            sample_sets = self.sample_sets()["sample_set"].tolist()

        elif isinstance(sample_sets, str):

            if sample_sets.startswith("1."):
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

    def _sample_metadata(self, *, sample_set):
        df = self._read_general_metadata(sample_set=sample_set)
        return df

    def sample_metadata(
        self,
        sample_sets=None,
        sample_query=None,
    ):
        """Access sample metadata for one or more sample sets.

        Parameters
        ----------
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "1229-VO-GH-DADZIE-VMF00095") or a list of
            sample set identifiers (e.g., ["1240-VO-CD-KOEKEMOER-VMF00099", "1240-VO-MZ-KOEKEMOER-VMF00101"]) or a
            release identifier (e.g., "1.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'funestus' and country == 'Burkina Faso'".

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

    def open_site_filters(self, mask):
        """Open site filters zarr.

        Parameters
        ----------
        mask : {"funestus"}
            Mask to use.

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        try:
            return self._cache_site_filters[mask]
        except KeyError:
            path = f"{self._base_path}/v1.0/site_filters/{self._site_filters_analysis}/{mask}/"
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_site_filters[mask] = root
            return root

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
            pos = self.snp_sites(region=region.contig, field="POS")
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
            Chromosome (e.g., "2RL"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2RL:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["2RL", "3RL"].
        mask : {"funestus"}
            Mask to use.
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

    def resolve_region(self, region):
        """Convert a genome region into a standard data structure.

        Parameters
        ----------
        region: str
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").

        Returns
        -------
        out : Region
            A named tuple with attributes contig, start and end.

        """

        # FIXME: self.resolve_region uses util.resolve_region
        return resolve_region(self, region)

    def open_snp_sites(self):
        """Open SNP sites zarr.

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        if self._cache_snp_sites is None:
            path = f"{self._base_path}/v1.0/snp_genotypes/all/sites/"
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
        inline_array=True,
        chunks="native",
    ):
        """Access SNP site data (positions and alleles).

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Chromosome (e.g., "2RL"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2RL:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["2RL", "3RL"].
        field : {"POS", "REF", "ALT"}
            Array to access.
        site_mask : {"funestus"}
            Site filters mask to apply.
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
            Chromosome (e.g., "2RL"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["2RL", "3RL"].
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "1229-VO-GH-DADZIE-VMF00095") or a list of
            sample set identifiers (e.g., ["1240-VO-CD-KOEKEMOER-VMF00099", "1240-VO-MZ-KOEKEMOER-VMF00101"]) or a
            release identifier (e.g., "1.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'funestus' and country == 'Burkina Faso'".
        field : {"GT", "GQ", "AD", "MQ"}
            Array to access.
        site_mask : {"funestus"}
            Site filters mask to apply.
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
            path = f"{self._base_path}/{GENOME_ZARR_PATH}"
            store = init_zarr_store(fs=self._fs, path=path)
            self._cache_genome = zarr.open_consolidated(store=store)
        return self._cache_genome

    def genome_sequence(self, region, inline_array=True, chunks="native"):
        """Access the reference genome sequence.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Chromosome (e.g., "2RL"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2RL:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["2RL", "3RL"].
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

    def geneset(self, region=None, attributes=("ID", "Parent", "Name", "description")):
        """Access genome feature annotations (idAnoFuneDA-416_04).

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Chromosome (e.g., "2RL"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2RL:44989425-44998059") or a
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
        debug = self._log.debug

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

    def _site_mask_ids(self):
        if self._site_filters_analysis == "dt_20200416":
            return ["funestus"]
        elif self._site_filters_analysis == "sc_20220908":
            return ["funestus"]
        else:
            raise ValueError

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
        inline_array=True,
        chunks="native",
    ):
        """Access SNP sites, site filters and genotype calls.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Chromosome (e.g., "2RL"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["2RL", "3RL"].
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "1229-VO-GH-DADZIE-VMF00095") or a list of
            sample set identifiers (e.g., ["1240-VO-CD-KOEKEMOER-VMF00099", "1240-VO-MZ-KOEKEMOER-VMF00101"]) or a
            release identifier (e.g., "1.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'funestus' and country == 'Burkina Faso'".
        site_mask : {"funestus"}
            Site filters mask to apply.
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr
            chunks. Also, can be a target size, e.g., '200 MiB'.

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
            df_samples = self.sample_metadata(sample_sets=sample_sets)
            loc_samples = df_samples.eval(sample_query).values
            if np.count_nonzero(loc_samples) == 0:
                raise ValueError(f"No samples found for query {sample_query!r}")
            ds = ds.isel(samples=loc_samples)

        return ds
