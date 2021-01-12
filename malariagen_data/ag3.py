import pandas
from fsspec.core import url_to_fs
import zarr
import dask.array as da
import numpy as np
from .util import read_gff3, unpack_gff3_attributes, SafeStore


public_releases = ("v3",)


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

    def __init__(self, url, **kwargs):

        # special case Google Cloud Storage, use anonymous access, avoids a delay
        if url.startswith("gs://") or url.startswith("gcs://"):
            kwargs.setdefault("token", "anon")

        # process the url using fsspec
        pre = kwargs.pop("pre", False)
        fs, path = url_to_fs(url, **kwargs)
        self.fs = fs
        self.path = path

        # discover which releases are available
        sub_dirs = [p.split("/")[-1] for p in self.fs.ls(self.path)]
        releases = [d for d in sub_dirs if d.startswith("v3")]
        if not pre:
            releases = [d for d in releases if d in public_releases]
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
        self._cache_geneset = None

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
            df["species"] = np.array([np.nan] * len(df), dtype=object)
            loc = df["species_gambcolu_arabiensis"].values == "arabiensis"
            df.loc[loc, "species"] = "arabiensis"
            loc = df["species_gambcolu_arabiensis"].values == "intermediate"
            df.loc[loc, "species"] = "intermediate_arabiensis_gambiae"
            loc = (df["species_gambcolu_arabiensis"].values == "gamb_colu") & (
                df["species_gambiae_coluzzii"].values == "gambiae"
            )
            df.loc[loc, "species"] = "gambiae"
            loc = (df["species_gambcolu_arabiensis"].values == "gamb_colu") & (
                df["species_gambiae_coluzzii"].values == "coluzzii"
            )
            df.loc[loc, "species"] = "coluzzii"
            loc = (df["species_gambcolu_arabiensis"].values == "gamb_colu") & (
                df["species_gambiae_coluzzii"].values == "intermediate"
            )
            df.loc[loc, "species"] = "intermediate_gambiae_coluzzii"

            self._cache_species_calls[key] = df
            return df

    def _prep_cohort(self, *, cohort):
        if cohort == "v3_wild":
            # convenience, special case to exclude crosses
            cohort = self.v3_wild

        elif isinstance(cohort, str) and cohort.startswith("v3"):
            # convenience, can use a release identifier to denote all sample sets
            # in a release
            cohort = self.sample_sets(release=cohort)["sample_set"].tolist()

        if not isinstance(cohort, (str, list, tuple)):
            raise TypeError(f"Invalid cohort: {cohort!r}")

        return cohort

    def species_calls(self, cohort="v3_wild", analysis="20200422", method="aim"):
        """Access species calls for one or more sample sets.

        Parameters
        ----------
        cohort : str or list of str
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

        cohort = self._prep_cohort(cohort=cohort)

        if isinstance(cohort, str):
            # assume single sample set
            df = self._read_species_calls(
                sample_set=cohort, analysis=analysis, method=method
            )

        else:
            # concatenate multiple sample sets
            dfs = [
                self.species_calls(cohort=c, analysis=analysis, method=method)
                for c in cohort
            ]
            df = pandas.concat(dfs, axis=0, sort=False).reset_index(drop=True)

        return df

    def sample_metadata(self, cohort="v3_wild", species_calls=("20200422", "aim")):
        """Access sample metadata for one or more sample sets.

        Parameters
        ----------
        cohort : str or list of str
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "v3") or a list of release identifiers.
        species_calls : (str, str), optional
            Include species calls in metadata.

        Returns
        -------
        df : pandas.DataFrame

        """

        cohort = self._prep_cohort(cohort=cohort)

        if isinstance(cohort, str):
            # assume single sample set
            df = self._read_general_metadata(sample_set=cohort)
            if species_calls is not None:
                analysis, method = species_calls
                df_species = self._read_species_calls(
                    sample_set=cohort, analysis=analysis, method=method
                )
                df = df.merge(df_species, on="sample_id", sort=False)

        else:
            # concatenate multiple sample sets
            dfs = [
                self.sample_metadata(cohort=c, species_calls=species_calls)
                for c in cohort
            ]
            df = pandas.concat(dfs, axis=0, sort=False).reset_index(drop=True)

        return df

    def _open_site_filters(self, *, mask, analysis):
        key = mask, analysis
        try:
            return self._cache_site_filters[key]
        except KeyError:
            path = f"{self.path}/v3/site_filters/{analysis}/{mask}/"
            store = SafeStore(self.fs.get_mapper(path))
            root = zarr.open_consolidated(store=store)
            self._cache_site_filters[key] = root
            return root

    def site_filters(self, seq_id, mask, field="filter_pass", analysis="dt_20200416"):
        """Access SNP site filters.

        Parameters
        ----------
        seq_id : str
            Chromosome arm, e.g., "3R".
        mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Mask to use.
        field : str, optional
            Array to access.
        analysis : str, optional
            Site filters analysis version.

        Returns
        -------
        d : dask.array.Array

        """

        root = self._open_site_filters(mask=mask, analysis=analysis)
        z = root[seq_id]["variants"][field]
        d = da.from_array(z, chunks=z.chunks)
        return d

    def _open_snp_sites(self):
        if self._cache_snp_sites is None:
            path = f"{self.path}/v3/snp_genotypes/all/sites/"
            store = SafeStore(self.fs.get_mapper(path))
            root = zarr.open_consolidated(store=store)
            self._cache_snp_sites = root
        return self._cache_snp_sites

    def snp_sites(self, seq_id, field=None, site_mask=None, site_filters="dt_20200416"):
        """Access SNP site data (positions and alleles).

        Parameters
        ----------
        seq_id : str
            Chromosome arm, e.g., "3R".
        field : {"POS", "REF", "ALT"}, optional
            Array to access. If not provided, all three arrays will be returned as a tuple.
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_filters : str
            Site filters analysis version.

        Returns
        -------
        d : dask.array.Array or tuple of dask.array.Array

        """

        if field is None:
            # return POS, REF, ALT
            return tuple(
                self.snp_sites(
                    seq_id=seq_id,
                    field=f,
                    site_mask=site_mask,
                    site_filters=site_filters,
                )
                for f in ("POS", "REF", "ALT")
            )

        root = self._open_snp_sites()
        z = root[seq_id]["variants"][field]
        d = da.from_array(z, chunks=z.chunks)

        if site_mask is not None:
            filter_pass = self.site_filters(
                seq_id=seq_id, mask=site_mask, analysis=site_filters
            ).compute()
            d = da.compress(filter_pass, d, axis=0)

        return d

    def _open_snp_genotypes(self, *, sample_set):
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
        seq_id,
        cohort="v3_wild",
        field="GT",
        site_mask=None,
        site_filters="dt_20200416",
    ):
        """Access SNP genotypes and associated data.

        Parameters
        ----------
        seq_id : str
            Chromosome arm, e.g., "3R".
        cohort : str or list of str
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of sample set
            identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a release identifier (e.g.,
            "v3") or a list of release identifiers.
        field : {"GT", "GQ", "AD", "MQ"}
            Array to access.
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_filters : str, optional
            Site filters analysis version.

        Returns
        -------
        d : dask.array.Array

        """

        cohort = self._prep_cohort(cohort=cohort)

        if isinstance(cohort, str):
            # single sample set
            root = self._open_snp_genotypes(sample_set=cohort)
            z = root[seq_id]["calldata"][field]
            d = da.from_array(z, chunks=z.chunks)

        else:
            # concatenate multiple sample sets
            ds = [
                self.snp_genotypes(seq_id=seq_id, cohort=c, field=field) for c in cohort
            ]
            d = da.concatenate(ds, axis=1)

        if site_mask is not None:
            filter_pass = self.site_filters(
                seq_id=seq_id, mask=site_mask, analysis=site_filters
            ).compute()
            d = da.compress(filter_pass, d, axis=0)

        return d

    def _open_genome(self):
        if self._cache_genome is None:
            path = f"{self.path}/reference/genome/agamp4/Anopheles-gambiae-PEST_CHROMOSOMES_AgamP4.zarr"
            store = SafeStore(self.fs.get_mapper(path))
            self._cache_genome = zarr.open_consolidated(store=store)
        return self._cache_genome

    def genome_sequence(self, seq_id):
        """Access the reference genome sequence.

        Parameters
        ----------
        seq_id : str
            Chromosome arm, e.g., "3R".

        Returns
        -------
        d : dask.array.Array

        """
        genome = self._open_genome()
        z = genome[seq_id]
        d = da.from_array(z, chunks=z.chunks)
        return d

    def _read_geneset(self):
        if self._cache_geneset is None:
            path = f"{self.path}/reference/genome/agamp4/Anopheles-gambiae-PEST_BASEFEATURES_AgamP4.12.gff3.gz"
            with self.fs.open(path, mode="rb") as f:
                self._cache_geneset = read_gff3(f, compression="gzip")
        return self._cache_geneset

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

        df = self._read_geneset()
        if attributes is not None:
            df = unpack_gff3_attributes(df, attributes=attributes)

        return df

    def is_accessible(self, seq_id, site_mask, site_filters="dt_20200416"):
        """Compute genome accessibility array.

        Parameters
        ----------
        seq_id : str
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
        seq_length = self.genome_sequence(seq_id).shape[0]

        # setup output
        is_accessible = np.zeros(seq_length, dtype=bool)

        # access positions
        pos = self.snp_sites(seq_id, field="POS").compute()

        # access site filters
        filter_pass = self.site_filters(
            seq_id, mask=site_mask, analysis=site_filters
        ).compute()

        # assign values from site filters
        is_accessible[pos - 1] = filter_pass

        return is_accessible
