import pandas
from fsspec.core import url_to_fs
import zarr
import dask.array as da
import numpy as np
import xarray
from .util import (
    read_gff3,
    unpack_gff3_attributes,
    SafeStore,
    from_zarr,
    dask_compress_dataset,
)


DIM_VARIANT = "variants"
DIM_ALLELE = "alleles"
DIM_SAMPLE = "samples"
DIM_PLOIDY = "ploidy"


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

    public_releases = ("v3",)
    contigs = ("2R", "2L", "3R", "3L", "X")

    def __init__(self, url, **kwargs):

        # special case Google Cloud Storage, use anonymous access, avoids a delay
        if url.startswith("gs://") or url.startswith("gcs://"):
            kwargs.setdefault("token", "anon")

        # process the url using fsspec
        pre = kwargs.pop("pre", False)
        fs, path = url_to_fs(url, **kwargs)
        self.fs = fs
        # path compatibility, fsspec/gcsfs behaviour varies between version
        while path.endswith("/"):
            path = path[:-1]
        self.path = path

        # discover which releases are available
        sub_dirs = [p.split("/")[-1] for p in self.fs.ls(self.path)]
        releases = [d for d in sub_dirs if d.startswith("v3")]
        if not pre:
            releases = [d for d in releases if d in self.public_releases]
        if len(releases) == 0:
            raise ValueError(f"No releases found at location {url!r}")
        self._releases = releases

        # setup caches
        self._cache_sample_sets = dict()
        self._cache_general_metadata = dict()
        self._cache_species_calls = dict()
        self._cache_site_filters = dict()
        self._cache_snp_sites = None
        self._cache_snp_genotypes = dict()
        self._cache_genome = None
        self._cache_geneset = dict()
        self._cache_cross_metadata = None
        self._cache_site_annotations = None

    def sample_sets(self, release="v3"):
        """Access the manifest of sample sets.

        Parameters
        ----------
        release : str
            Release identifier. Give "v3" to access the Ag1000G phase 3 data release.

        Returns
        -------
        df : pandas.DataFrame

        """

        if release not in self._releases:
            raise ValueError(f"Release not available: {release!r}")

        try:
            return self._cache_sample_sets[release]

        except KeyError:
            path = f"{self.path}/{release}/manifest.tsv"
            with self.fs.open(path) as f:
                df = pandas.read_csv(f, sep="\t", na_values="")
            df["release"] = release
            self._cache_sample_sets[release] = df
            return df

    @property
    def v3_wild(self):
        return [
            x
            for x in self.sample_sets(release="v3")["sample_set"].tolist()
            if x != "AG1000G-X"
        ]

    def _lookup_release(self, *, sample_set):
        # find which release this sample set was included in
        for release in self._releases:
            df_sample_sets = self.sample_sets(release=release)
            if sample_set in df_sample_sets["sample_set"].tolist():
                return release
        raise ValueError(f"No release found for sample set {sample_set!r}")

    def _read_general_metadata(self, *, sample_set):
        """Read metadata for a single sample set."""
        try:
            return self._cache_general_metadata[sample_set]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            path = (
                f"{self.path}/{release}/metadata/general/{sample_set}/samples.meta.csv"
            )
            with self.fs.open(path) as f:
                df = pandas.read_csv(f, na_values="")

            # add a couple of columns for convenience
            df["sample_set"] = sample_set
            df["release"] = release

            self._cache_general_metadata[sample_set] = df
            return df

    def _read_species_calls(self, *, sample_set, analysis, method):
        """Read species calls for a single sample set."""
        key = (sample_set, analysis, method)
        try:
            return self._cache_species_calls[key]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            path = (
                f"{self.path}/{release}/metadata/species_calls_{analysis}"
                f"/{sample_set}/samples.species_{method}.csv"
            )
            with self.fs.open(path) as f:
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

            self._cache_species_calls[key] = df
            return df

    def _prep_sample_sets_arg(self, *, sample_sets):
        if sample_sets == "v3_wild":
            # convenience, special case to exclude crosses
            sample_sets = self.v3_wild

        elif isinstance(sample_sets, str) and sample_sets.startswith("v3"):
            # convenience, can use a release identifier to denote all sample sets
            # in a release
            sample_sets = self.sample_sets(release=sample_sets)["sample_set"].tolist()

        if not isinstance(sample_sets, (str, list, tuple)):
            raise TypeError(f"Invalid sample_sets: {sample_sets!r}")

        return sample_sets

    def species_calls(self, sample_sets="v3_wild", analysis="20200422", method="aim"):
        """Access species calls for one or more sample sets.

        Parameters
        ----------
        sample_sets : str or list of str
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"] or a release identifier (e.g.,
            "v3") or a list of release identifiers.
        analysis : str
            Species calling analysis version.
        method : str
            Species calling method; "aim" is ancestry informative markers, "pca" is principal
            components analysis.

        Returns
        -------
        df : pandas.DataFrame

        """

        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)

        if isinstance(sample_sets, str):
            # assume single sample set
            df = self._read_species_calls(
                sample_set=sample_sets, analysis=analysis, method=method
            )

        else:
            # concatenate multiple sample sets
            dfs = [
                self.species_calls(sample_sets=c, analysis=analysis, method=method)
                for c in sample_sets
            ]
            df = pandas.concat(dfs, axis=0, sort=False).reset_index(drop=True)

        return df

    def sample_metadata(self, sample_sets="v3_wild", species_calls=("20200422", "aim")):
        """Access sample metadata for one or more sample sets.

        Parameters
        ----------
        sample_sets : str or list of str
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "v3") or a list of release identifiers.
        species_calls : (str, str), optional
            Include species calls in metadata.

        Returns
        -------
        df : pandas.DataFrame

        """

        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)

        if isinstance(sample_sets, str):
            # assume single sample set
            df = self._read_general_metadata(sample_set=sample_sets)
            if species_calls is not None:
                analysis, method = species_calls
                df_species = self._read_species_calls(
                    sample_set=sample_sets, analysis=analysis, method=method
                )
                df = df.merge(df_species, on="sample_id", sort=False)

        else:
            # concatenate multiple sample sets
            dfs = [
                self.sample_metadata(sample_sets=c, species_calls=species_calls)
                for c in sample_sets
            ]
            df = pandas.concat(dfs, axis=0, sort=False).reset_index(drop=True)

        return df

    def open_site_filters(self, mask, analysis="dt_20200416"):
        """Open site filters zarr.

        Parameters
        ----------
        mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Mask to use.
        analysis : str, optional
            Site filters analysis version.

        Returns
        -------
        genome : zarr.hierarchy.Group

        """
        key = mask, analysis
        try:
            return self._cache_site_filters[key]
        except KeyError:
            path = f"{self.path}/v3/site_filters/{analysis}/{mask}/"
            store = SafeStore(self.fs.get_mapper(path))
            root = zarr.open_consolidated(store=store)
            self._cache_site_filters[key] = root
            return root

    def site_filters(
        self,
        contig,
        mask,
        field="filter_pass",
        analysis="dt_20200416",
        inline_array=True,
    ):
        """Access SNP site filters.

        Parameters
        ----------
        contig : str
            Chromosome arm, e.g., "3R".
        mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Mask to use.
        field : str, optional
            Array to access.
        analysis : str, optional
            Site filters analysis version.
        inline_array : bool, optional
            Passed through to dask.from_array().

        Returns
        -------
        d : dask.array.Array

        """

        root = self.open_site_filters(mask=mask, analysis=analysis)
        z = root[contig]["variants"][field]
        d = from_zarr(z, inline_array=inline_array)
        return d

    def open_snp_sites(self):
        """Open SNP sites zarr.

        Returns
        -------
        genome : zarr.hierarchy.Group

        """
        if self._cache_snp_sites is None:
            path = f"{self.path}/v3/snp_genotypes/all/sites/"
            store = SafeStore(self.fs.get_mapper(path))
            root = zarr.open_consolidated(store=store)
            self._cache_snp_sites = root
        return self._cache_snp_sites

    def snp_sites(
        self,
        contig,
        field=None,
        site_mask=None,
        site_filters="dt_20200416",
        inline_array=True,
    ):
        """Access SNP site data (positions and alleles).

        Parameters
        ----------
        contig : str
            Chromosome arm, e.g., "3R".
        field : {"POS", "REF", "ALT"}, optional
            Array to access. If not provided, all three arrays POS, REF, ALT will be returned as a tuple.
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_filters : str
            Site filters analysis version.
        inline_array : bool, optional
            Passed through to dask.array.from_array().

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

        else:
            root = self.open_snp_sites()
            z = root[contig]["variants"][field]
            ret = from_zarr(z, inline_array=inline_array)

        if site_mask is not None:
            loc_sites = self.site_filters(
                contig=contig, mask=site_mask, analysis=site_filters
            )
            if isinstance(ret, tuple):
                ret = tuple(da.compress(loc_sites, d, axis=0) for d in ret)
                for a in ret:
                    a.compute_chunk_sizes()
            else:
                ret = da.compress(loc_sites, ret, axis=0)
                ret.compute_chunk_sizes()

        return ret

    def open_snp_genotypes(self, sample_set):
        """Open SNP genotypes zarr.

        Parameters
        ----------
        sample_set : str

        Returns
        -------
        genome : zarr.hierarchy.Group

        """
        try:
            return self._cache_snp_genotypes[sample_set]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            path = f"{self.path}/{release}/snp_genotypes/all/{sample_set}/"
            store = SafeStore(self.fs.get_mapper(path))
            root = zarr.open_consolidated(store=store)
            self._cache_snp_genotypes[sample_set] = root
            return root

    def snp_genotypes(
        self,
        contig,
        sample_sets="v3_wild",
        field="GT",
        site_mask=None,
        site_filters="dt_20200416",
        inline_array=True,
    ):
        """Access SNP genotypes and associated data.

        Parameters
        ----------
        contig : str
            Chromosome arm, e.g., "3R".
        sample_sets : str or list of str
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

        Returns
        -------
        d : dask.array.Array

        """

        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)

        if isinstance(sample_sets, str):
            # single sample set
            root = self.open_snp_genotypes(sample_set=sample_sets)
            z = root[contig]["calldata"][field]
            d = from_zarr(z, inline_array=inline_array)

        else:
            # concatenate multiple sample sets
            ds = [
                self.snp_genotypes(contig=contig, sample_sets=c, field=field)
                for c in sample_sets
            ]
            d = da.concatenate(ds, axis=1)

        if site_mask is not None:
            loc_sites = self.site_filters(
                contig=contig, mask=site_mask, analysis=site_filters
            )
            d = da.compress(loc_sites, d, axis=0)
            d.compute_chunk_sizes()

        return d

    def open_genome(self):
        """Open the reference genome zarr.

        Returns
        -------
        genome : zarr.hierarchy.Group

        """
        if self._cache_genome is None:
            path = f"{self.path}/reference/genome/agamp4/Anopheles-gambiae-PEST_CHROMOSOMES_AgamP4.zarr"
            store = SafeStore(self.fs.get_mapper(path))
            self._cache_genome = zarr.open_consolidated(store=store)
        return self._cache_genome

    def genome_sequence(self, contig, inline_array=True):
        """Access the reference genome sequence.

        Parameters
        ----------
        contig : str
            Chromosome arm, e.g., "3R".
        inline_array : bool, optional
            Passed through to dask.array.from_array().

        Returns
        -------
        d : dask.array.Array

        """
        genome = self.open_genome()
        z = genome[contig]
        d = from_zarr(z, inline_array=inline_array)
        return d

    def geneset(self, attributes=("ID", "Parent", "Name")):
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
            path = f"{self.path}/reference/genome/agamp4/Anopheles-gambiae-PEST_BASEFEATURES_AgamP4.12.gff3.gz"
            with self.fs.open(path, mode="rb") as f:
                df = read_gff3(f, compression="gzip")
            if attributes is not None:
                df = unpack_gff3_attributes(df, attributes=attributes)
            self._cache_geneset[attributes] = df

        return df

    def is_accessible(self, contig, site_mask, site_filters="dt_20200416"):
        """Compute genome accessibility array.

        Parameters
        ----------
        contig : str
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

    def cross_metadata(self):
        """Load a dataframe containing metadata about samples in colony crosses, including
        which samples are parents or progeny in which crosses.

        Returns
        -------
        df : pandas.DataFrame

        """

        if self._cache_cross_metadata is None:

            path = f"{self.path}/v3/metadata/crosses/crosses.fam"
            fam_names = [
                "cross",
                "sample_id",
                "father_id",
                "mother_id",
                "sex",
                "phenotype",
            ]
            with self.fs.open(path) as f:
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
        if self._cache_site_annotations is None:
            path = f"{self.path}/reference/genome/agamp4/Anopheles-gambiae-PEST_SEQANNOTATION_AgamP4.12.zarr"
            self._cache_site_annotations = zarr.open_consolidated(
                self.fs.get_mapper(path)
            )
        return self._cache_site_annotations

    def site_annotations(
        self,
        contig,
        field,
        site_mask=None,
        site_filters="dt_20200416",
        inline_array=True,
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

        Returns
        -------
        d : dask.Array

        """

        # access the array of values for all genome positions
        root = self.open_site_annotations()
        d = from_zarr(root[field][contig], inline_array=inline_array)

        # access and subset to SNP positions
        pos = self.snp_sites(
            contig=contig, field="POS", site_mask=site_mask, site_filters=site_filters
        )
        d = da.take(d, pos - 1)

        return d

    def _snp_calls_dataset(self, contig, sample_set, site_filters, inline_array):

        coords = dict()
        data_vars = dict()

        # variant arrays
        sites_root = self.open_snp_sites()

        # variant_position
        pos_z = sites_root[contig]["variants"]["POS"]
        variant_position = from_zarr(pos_z, inline_array=inline_array)
        coords["variant_position"] = [DIM_VARIANT], variant_position

        # variant_allele
        ref_z = sites_root[contig]["variants"]["REF"]
        ref = from_zarr(ref_z, inline_array=inline_array)
        alt_z = sites_root[contig]["variants"]["ALT"]
        alt = from_zarr(alt_z, inline_array=inline_array)
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
            z = filters_root[contig]["variants"]["filter_pass"]
            d = from_zarr(z, inline_array=inline_array)
            data_vars[f"variant_filter_pass_{mask}"] = [DIM_VARIANT], d

        # call arrays
        calls_root = self.open_snp_genotypes(sample_set=sample_set)
        gt_z = calls_root[contig]["calldata"]["GT"]
        call_genotype = from_zarr(gt_z, inline_array=inline_array)
        gq_z = calls_root[contig]["calldata"]["GQ"]
        call_gq = from_zarr(gq_z, inline_array=inline_array)
        ad_z = calls_root[contig]["calldata"]["AD"]
        call_ad = from_zarr(ad_z, inline_array=inline_array)
        mq_z = calls_root[contig]["calldata"]["MQ"]
        call_mq = from_zarr(mq_z, inline_array=inline_array)
        data_vars["call_genotype"] = (
            [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY],
            call_genotype,
        )
        data_vars["call_GQ"] = ([DIM_VARIANT, DIM_SAMPLE], call_gq)
        data_vars["call_MQ"] = ([DIM_VARIANT, DIM_SAMPLE], call_mq)
        data_vars["call_AD"] = ([DIM_VARIANT, DIM_SAMPLE, DIM_ALLELE], call_ad)

        # sample arrays
        z = calls_root["samples"]
        sample_id = from_zarr(z, inline_array=inline_array)
        coords["sample_id"] = [DIM_SAMPLE], sample_id

        # setup attributes
        attrs = {"contigs": self.contigs}

        # create a dataset
        ds = xarray.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        return ds

    def snp_calls(
        self,
        contig,
        sample_sets="v3_wild",
        site_mask=None,
        site_filters="dt_20200416",
        inline_array=True,
    ):
        """Access SNP sites, site filters and genotype calls.

        Parameters
        ----------
        contig : str
            Chromosome arm, e.g., "3R".
        sample_sets : str or list of str
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "v3") or a list of release identifiers.
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_filters : str
            Site filters analysis version.
        inline_array : bool, optional
            Passed through to dask.array.from_array().

        Returns
        -------
        ds : xarray.Dataset

        """

        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)

        if isinstance(sample_sets, str):

            # single sample set requested
            ds = self._snp_calls_dataset(
                contig=contig,
                sample_set=sample_sets,
                site_filters=site_filters,
                inline_array=inline_array,
            )

        else:

            # multiple sample sets requested, need to concatenate along samples dimension
            datasets = [
                self._snp_calls_dataset(
                    contig=contig,
                    sample_set=sample_set,
                    site_filters=site_filters,
                    inline_array=inline_array,
                )
                for sample_set in sample_sets
            ]
            ds = xarray.concat(
                datasets,
                dim=DIM_SAMPLE,
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
