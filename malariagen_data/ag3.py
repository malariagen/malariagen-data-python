from bisect import bisect_left, bisect_right

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

PUBLIC_RELEASES = ("v3",)
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
    Access data from Google Cloud Storage:

        >>> import malariagen_data
        >>> ag3 = malariagen_data.Ag3("gs://vo_agam_release/")

    Access data downloaded to a local file system:

        >>> ag3 = malariagen_data.Ag3("/local/path/to/vo_agam_release/")

    """

    contigs = CONTIGS

    def __init__(self, url, **kwargs):

        self._pre = kwargs.pop("pre", False)

        # setup filesystem
        self._fs, self._path = init_filesystem(url, **kwargs)

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

    @property
    def releases(self):
        if self._cache_releases is None:
            if self._pre:
                # discover which releases are available
                sub_dirs = [p.split("/")[-1] for p in self._fs.ls(self._path)]
                releases = tuple(
                    sorted(
                        [
                            d
                            for d in sub_dirs
                            if d.startswith("v3")
                            and self._fs.exists(f"{self._path}/{d}/manifest.tsv")
                        ]
                    )
                )
                if len(releases) == 0:
                    raise ValueError("No releases found.")
                self._cache_releases = releases
            else:
                self._cache_releases = PUBLIC_RELEASES
        return self._cache_releases

    def sample_sets(self, release=None):
        """Access a dataframe of sample sets.

        Parameters
        ----------
        release : str, optional
            Release identifier. Give "v3" to access the Ag1000G phase 3 data release.

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
                path = f"{self._path}/{release}/manifest.tsv"
                with self._fs.open(path) as f:
                    df = pandas.read_csv(f, sep="\t", na_values="")
                df["release"] = release
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
        return [
            x
            for x in self.sample_sets(release="v3")["sample_set"].tolist()
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
            path = (
                f"{self._path}/{release}/metadata/general/{sample_set}/samples.meta.csv"
            )
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
            if analysis == "aim_20200422":
                path = f"{self._path}/{release}/metadata/species_calls_20200422/{sample_set}/samples.species_aim.csv"
            elif analysis == "pca_20200422":
                path = f"{self._path}/{release}/metadata/species_calls_20200422/{sample_set}/samples.species_pca.csv"
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

            if sample_sets == "v3_wild":
                # convenience, special case to exclude crosses
                sample_sets = self.v3_wild

            elif sample_sets.startswith("v3"):
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

    def species_calls(self, sample_sets=None, analysis=DEFAULT_SPECIES_ANALYSIS):
        """Access species calls for one or more sample sets.

        Parameters
        ----------
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"] or a release identifier (e.g.,
            "v3") or a list of release identifiers.
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

    def _sample_metadata(self, *, sample_set, species_calls):
        df = self._read_general_metadata(sample_set=sample_set)
        if species_calls is not None:
            df_species = self._read_species_calls(
                sample_set=sample_set, analysis=species_calls
            )
            df = df.merge(df_species, on="sample_id", sort=False)
        return df

    def sample_metadata(self, sample_sets=None, species_calls=DEFAULT_SPECIES_ANALYSIS):
        """Access sample metadata for one or more sample sets.

        Parameters
        ----------
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "v3") or a list of release identifiers.
        species_calls : {"aim_20200422", "pca_20200422"}, optional
            Include species calls in metadata.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of sample metadata, one row per sample.

        """

        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)

        # concatenate multiple sample sets
        dfs = [
            self._sample_metadata(sample_set=s, species_calls=species_calls)
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
            path = f"{self._path}/v3/site_filters/{analysis}/{mask}/"
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_site_filters[key] = root
            return root

    def site_filters(
        self,
        contig,
        mask,
        field="filter_pass",
        analysis=DEFAULT_SITE_FILTERS_ANALYSIS,
        inline_array=True,
        chunks="native",
    ):
        """Access SNP site filters.

        Parameters
        ----------
        contig : str or list of str
            Chromosome arm, e.g., "3R". Multiple values can be provided as a list,
            in which case data will be concatenated, e.g., ["3R", "3L"].
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

        if isinstance(contig, (list, tuple)):
            return da.concatenate(
                [
                    self.site_filters(
                        contig=c,
                        mask=mask,
                        field=field,
                        analysis=analysis,
                        inline_array=inline_array,
                        chunks=chunks,
                    )
                    for c in contig
                ]
            )
        else:
            root = self.open_site_filters(mask=mask, analysis=analysis)
            z = root[contig]["variants"][field]
            d = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
            return d

    def open_snp_sites(self):
        """Open SNP sites zarr.

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        if self._cache_snp_sites is None:
            path = f"{self._path}/v3/snp_genotypes/all/sites/"
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_snp_sites = root
        return self._cache_snp_sites

    def snp_sites(
        self,
        contig,
        field=None,
        site_mask=None,
        site_filters=DEFAULT_SITE_FILTERS_ANALYSIS,
        inline_array=True,
        chunks="native",
    ):
        """Access SNP site data (positions and alleles).

        Parameters
        ----------
        contig : str or list of str
            Chromosome arm, e.g., "3R". Multiple values can be provided as a list,
            in which case data will be concatenated, e.g., ["3R", "3L"].
        field : {"POS", "REF", "ALT"}, optional
            Array to access. If not provided, all three arrays POS, REF, ALT will be returned as a
            tuple.
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_filters : str
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
                self.snp_sites(contig=contig, field=f, site_mask=None)
                for f in ("POS", "REF", "ALT")
            )

        elif isinstance(contig, (tuple, list)):
            # concatenate
            ret = da.concatenate(
                [self.snp_sites(contig=c, field=field, site_mask=None) for c in contig]
            )

        else:
            root = self.open_snp_sites()
            z = root[contig]["variants"][field]
            ret = da_from_zarr(z, inline_array=inline_array, chunks=chunks)

        if site_mask is not None:
            loc_sites = self.site_filters(
                contig=contig, mask=site_mask, analysis=site_filters
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
            path = f"{self._path}/{release}/snp_genotypes/all/{sample_set}/"
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_snp_genotypes[sample_set] = root
            return root

    def _snp_genotypes(self, *, contig, sample_set, field, inline_array, chunks):
        # single contig, single sample set
        root = self.open_snp_genotypes(sample_set=sample_set)
        z = root[contig]["calldata"][field]
        d = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        return d

    def snp_genotypes(
        self,
        contig,
        sample_sets=None,
        field="GT",
        site_mask=None,
        site_filters=DEFAULT_SITE_FILTERS_ANALYSIS,
        inline_array=True,
        chunks="native",
    ):
        """Access SNP genotypes and associated data.

        Parameters
        ----------
        contig : str or list of str
            Chromosome arm, e.g., "3R". Multiple values can be provided as a list,
            in which case data will be concatenated, e.g., ["3R", "3L"].
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "v3") or a list of release identifiers.
        field : {"GT", "GQ", "AD", "MQ"}
            Array to access.
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_filters : str, optional
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
        if isinstance(contig, str):
            contig = [contig]

        # concatenate multiple sample sets and/or contigs
        d = da.concatenate(
            [
                da.concatenate(
                    [
                        self._snp_genotypes(
                            contig=c,
                            sample_set=s,
                            field=field,
                            inline_array=inline_array,
                            chunks=chunks,
                        )
                        for s in sample_sets
                    ],
                    axis=1,
                )
                for c in contig
            ],
            axis=0,
        )

        # apply site filters if requested
        if site_mask is not None:
            loc_sites = self.site_filters(
                contig=contig, mask=site_mask, analysis=site_filters
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
            path = f"{self._path}/{GENOME_ZARR_PATH}"
            store = init_zarr_store(fs=self._fs, path=path)
            self._cache_genome = zarr.open_consolidated(store=store)
        return self._cache_genome

    def genome_sequence(self, contig, inline_array=True, chunks="native"):
        """Access the reference genome sequence.

        Parameters
        ----------
        contig : str
            Chromosome arm, e.g., "3R".
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
        z = genome[contig]
        d = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        return d

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
            path = f"{self._path}/{GENESET_GFF3_PATH}"
            with self._fs.open(path, mode="rb") as f:
                df = read_gff3(f, compression="gzip")
            if attributes is not None:
                df = unpack_gff3_attributes(df, attributes=attributes)
            self._cache_geneset[attributes] = df

        return df

    def is_accessible(
        self, contig, site_mask, site_filters=DEFAULT_SITE_FILTERS_ANALYSIS
    ):
        """Compute genome accessibility array.

        Parameters
        ----------
        contig : str or list of str
            Chromosome arm, e.g., "3R".
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_filters : str, optional
            Site filters analysis version.

        Returns
        -------
        a : numpy.ndarray

        """

        # determine contig sequence length
        seq_length = self.genome_sequence(contig).shape[0]

        # setup output
        is_accessible = np.zeros(seq_length, dtype=bool)

        # access positions
        pos = self.snp_sites(contig, field="POS").compute()

        # access site filters
        filter_pass = self.site_filters(
            contig, mask=site_mask, analysis=site_filters
        ).compute()

        # assign values from site filters
        is_accessible[pos - 1] = filter_pass

        return is_accessible

    def _site_mask_ids(self, *, site_filters):
        if site_filters == "dt_20200416":
            return "gamb_colu_arab", "gamb_colu", "arab"
        else:
            raise ValueError

    def _snp_df(self, *, transcript, site_filters):
        """Set up a dataframe with SNP site and filter columns."""

        # get feature direct from geneset
        gs = self.geneset()
        feature = gs[gs["ID"] == transcript].squeeze()
        contig = feature.contig

        # grab pos, ref and alt for chrom arm from snp_sites
        pos, ref, alt = self.snp_sites(contig=contig)

        # sites are dask arrays, turn pos into sorted index
        pos = allel.SortedIndex(pos.compute())

        # locate transcript range
        loc_feature = pos.locate_range(feature.start, feature.end)

        # dask compute on the sliced arrays to speed things up
        pos = pos[loc_feature]
        ref = ref[loc_feature].compute()
        alt = alt[loc_feature].compute()

        # access site filters
        filter_pass = dict()
        masks = self._site_mask_ids(site_filters=site_filters)
        for m in masks:
            x = self.site_filters(contig=feature.contig, mask=m, analysis=site_filters)
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
        df_snps = pandas.DataFrame(cols)

        return contig, loc_feature, df_snps

    def _annotator(self):
        # setup variant effect annotator
        if self._cache_annotator is None:
            self._cache_annotator = veff.Annotator(
                genome=self.open_genome(), geneset=self.geneset()
            )
        return self._cache_annotator

    def snp_effects(
        self, transcript, site_mask=None, site_filters=DEFAULT_SITE_FILTERS_ANALYSIS
    ):
        """Compute variant effects for a gene transcript.

        Parameters
        ----------
        transcript : str
            Gene transcript ID (AgamP4.12), e.g., "AGAP004707-RA".
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}, optional
            Site filters mask to apply.
        site_filters : str, optional
            Site filters analysis version.

        Returns
        -------
        df : pandas.DataFrame

        """

        # setup initial dataframe of SNPs
        _, _, df_snps = self._snp_df(transcript=transcript, site_filters=site_filters)

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
        self, *, cohorts, sample_sets, species_calls, cohorts_analysis
    ):

        # build cohort dictionary where key=cohort_id, value=loc_coh
        coh_dict = {}
        if isinstance(cohorts, dict):
            # get sample metadata
            df_meta = self.sample_metadata(
                sample_sets=sample_sets, species_calls=species_calls
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
        cohorts_analysis=DEFAULT_COHORTS_ANALYSIS,
        min_cohort_size=10,
        site_mask=None,
        site_filters=DEFAULT_SITE_FILTERS_ANALYSIS,
        species_calls=DEFAULT_SPECIES_ANALYSIS,
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
        cohorts_analysis : str
            Cohort analysis identifier (date of analysis), default is latest version.
        min_cohort_size : int
            Minimum cohort size, below which allele frequencies are not calculated for cohorts.
            Please note, NaNs will be returned for any cohorts with fewer samples than min_cohort_size,
            these can be removed from the output dataframe using pandas df.dropna(axis='columns').
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_filters : str, optional
            Site filters analysis version.
        species_calls : {"aim_20200422", "pca_20200422"}, optional
            Include species calls in metadata.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "v3") or a list of release identifiers.
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
        NaNs will be returned for any cohorts with fewer samples than min_cohort_size,
        these can be removed from the output dataframe using pandas df.dropna(axis='columns').

        """

        # setup initial dataframe of SNPs
        contig, loc_feature, df_snps = self._snp_df(
            transcript=transcript, site_filters=site_filters
        )

        # get genotypes
        gt = self.snp_genotypes(
            contig=contig,
            sample_sets=sample_sets,
            field="GT",
        )

        # slice to feature location
        gt = gt[loc_feature].compute()

        # build coh dict
        coh_dict = self._prep_cohorts_arg(
            cohorts=cohorts,
            sample_sets=sample_sets,
            species_calls=species_calls,
            cohorts_analysis=cohorts_analysis,
        )

        # count alleles
        for coh, loc_coh in coh_dict.items():
            n_samples = np.count_nonzero(loc_coh)
            if n_samples == 0:
                raise ValueError(f"no samples for cohort {coh!r}")
            if n_samples < min_cohort_size:
                df_snps[coh] = np.nan
            else:
                gt_coh = np.compress(loc_coh, gt, axis=1)
                # count alleles
                ac_coh = allel.GenotypeArray(gt_coh).count_alleles(max_allele=3)
                # compute allele frequencies
                af_coh = ac_coh.to_frequencies()
                # add column to dataframe
                df_snps[coh] = af_coh[:, 1:].flatten()

        # add max allele freq column
        df_snps["max_af"] = df_snps[list(coh_dict.keys())].max(axis=1)

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

            path = f"{self._path}/v3/metadata/crosses/crosses.fam"
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
            path = f"{self._path}/reference/genome/agamp4/Anopheles-gambiae-PEST_SEQANNOTATION_AgamP4.12.zarr"
            store = init_zarr_store(fs=self._fs, path=path)
            self._cache_site_annotations = zarr.open_consolidated(store=store)
        return self._cache_site_annotations

    def site_annotations(
        self,
        contig,
        field,
        site_mask=None,
        site_filters=DEFAULT_SITE_FILTERS_ANALYSIS,
        inline_array=True,
        chunks="native",
    ):
        """Load site annotations.

        Parameters
        ----------
        contig : str
            Chromosome arm, e.g., "3R".
        field : str
            One of "codon_degeneracy", "codon_nonsyn", "codon_position", "seq_cls",
            "seq_flen", "seq_relpos_start", "seq_relpos_stop".
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_filters : str
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
        d = da_from_zarr(root[field][contig], inline_array=inline_array, chunks=chunks)

        # access and subset to SNP positions
        pos = self.snp_sites(
            contig=contig, field="POS", site_mask=site_mask, site_filters=site_filters
        )
        d = da.take(d, pos - 1)

        return d

    def _snp_calls_dataset(
        self, *, contig, sample_set, site_filters, inline_array, chunks
    ):

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
            filters_root = self.open_site_filters(mask=mask, analysis=site_filters)
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
        data_vars["call_AD"] = ([DIM_VARIANT, DIM_SAMPLE, DIM_ALLELE], call_ad)

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
        contig,
        sample_sets=None,
        site_mask=None,
        site_filters=DEFAULT_SITE_FILTERS_ANALYSIS,
        inline_array=True,
        chunks="native",
    ):
        """Access SNP sites, site filters and genotype calls.

        Parameters
        ----------
        contig : str or list of str
            Chromosome arm, e.g., "3R". Multiple values can be provided as a list,
            in which case data will be concatenated, e.g., ["3R", "3L"].
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "v3") or a list of release identifiers.
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_filters : str
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
        if isinstance(contig, str):
            contig = [contig]

        # concatenate multiple sample sets and/or contigs
        ds = xarray.concat(
            [
                xarray.concat(
                    [
                        self._snp_calls_dataset(
                            contig=c,
                            sample_set=s,
                            site_filters=site_filters,
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
                for c in contig
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
            path = f"{self._path}/{release}/cnv/{sample_set}/hmm/zarr"
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
            "v3") or a list of release identifiers.
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
            path = f"{self._path}/{release}/cnv/{sample_set}/coverage_calls/{analysis}/zarr"
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
            path = f"{self._path}/{release}/cnv/{sample_set}/discordant_read_calls/zarr"
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
            "v3") or a list of release identifiers.
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
            "v3") or a list of release identifiers.

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
        cohorts_analysis=DEFAULT_COHORTS_ANALYSIS,
        min_cohort_size=10,
        species_calls=DEFAULT_SPECIES_ANALYSIS,
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
        cohorts_analysis : str
            Cohort analysis identifier (date of analysis), default is latest version.
        min_cohort_size : int
            Minimum cohort size, below which allele frequencies are not calculated for cohorts.
            Please note, NaNs will be returned for any cohorts with fewer samples than min_cohort_size,
            these can be removed from the output dataframe using pandas df.dropna(axis='columns').
        species_calls : {"aim_20200422", "pca_20200422"}, optional
            Include species calls in metadata.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "v3") or a list of release identifiers.

        Returns
        -------
        df : pandas.DataFrame

        Notes
        -----
        NaNs will be returned for any cohorts with fewer samples than min_cohort_size,
        these can be removed from the output dataframe using pandas df.dropna(axis='columns').

        """

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
            species_calls=species_calls,
            cohorts_analysis=cohorts_analysis,
        )

        # compute cohort frequencies
        freq_cols = dict()
        for coh, loc_samples in coh_dict.items():
            n_samples = np.count_nonzero(loc_samples)
            if n_samples == 0:
                raise ValueError(f"no samples for cohort {coh!r}")
            if n_samples < min_cohort_size:
                freq_cols[f"{coh}_amp"] = np.nan
                freq_cols[f"{coh}_del"] = np.nan
            else:
                is_amp_coh = np.compress(loc_samples, is_amp, axis=1)
                is_del_coh = np.compress(loc_samples, is_del, axis=1)
                amp_count_coh = np.sum(is_amp_coh, axis=1)
                del_count_coh = np.sum(is_del_coh, axis=1)
                amp_freq_coh = amp_count_coh / n_samples
                del_freq_coh = del_count_coh / n_samples
                freq_cols[f"{coh}_amp"] = amp_freq_coh
                freq_cols[f"{coh}_del"] = del_freq_coh

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
            path = f"{self._path}/{release}/snp_haplotypes/{sample_set}/{analysis}/zarr"
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
            path = f"{self._path}/v3/snp_haplotypes/sites/{analysis}/zarr"
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
            "v3") or a list of release identifiers.
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
            path = f"{self._path}/{release}/metadata/cohorts_{cohorts_analysis}/{sample_set}/samples.cohorts.csv"
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
            "v3") or a list of release identifiers.
        cohorts_analysis : str
            Cohort analysis identifier (date of analysis), default is latest version.

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
