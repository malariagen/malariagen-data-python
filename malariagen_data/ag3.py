import pandas
from fsspec.core import url_to_fs
import zarr
import dask.array as da
import numpy as np
import xarray
from .util import read_gff3, unpack_gff3_attributes, SafeStore


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
        self._cache_cnv_hmm = dict()
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

    def site_filters(self, contig, mask, field="filter_pass", analysis="dt_20200416"):
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

        Returns
        -------
        d : dask.array.Array

        """

        root = self.open_site_filters(mask=mask, analysis=analysis)
        z = root[contig]["variants"][field]
        d = da.from_array(z, chunks=z.chunks)
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

    def snp_sites(self, contig, field=None, site_mask=None, site_filters="dt_20200416"):
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
            ret = da.from_array(z, chunks=z.chunks)

        if site_mask is not None:
            filter_pass = self.site_filters(
                contig=contig, mask=site_mask, analysis=site_filters
            ).compute()
            if isinstance(ret, tuple):
                ret = tuple(da.compress(filter_pass, d, axis=0) for d in ret)
            else:
                ret = da.compress(filter_pass, ret, axis=0)

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

        Returns
        -------
        d : dask.array.Array

        """

        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)

        if isinstance(sample_sets, str):
            # single sample set
            root = self.open_snp_genotypes(sample_set=sample_sets)
            z = root[contig]["calldata"][field]
            d = da.from_array(z, chunks=z.chunks)

        else:
            # concatenate multiple sample sets
            ds = [
                self.snp_genotypes(contig=contig, sample_sets=c, field=field)
                for c in sample_sets
            ]
            d = da.concatenate(ds, axis=1)

        if site_mask is not None:
            filter_pass = self.site_filters(
                contig=contig, mask=site_mask, analysis=site_filters
            ).compute()
            d = da.compress(filter_pass, d, axis=0)

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

    def genome_sequence(self, contig):
        """Access the reference genome sequence.

        Parameters
        ----------
        contig : str
            Chromosome arm, e.g., "3R".

        Returns
        -------
        d : dask.array.Array

        """
        genome = self.open_genome()
        z = genome[contig]
        d = da.from_array(z, chunks=z.chunks)
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
        self, contig, field, site_mask=None, site_filters="dt_20200416"
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

        Returns
        -------
        d : dask.Array

        """

        # access the array of values for all genome positions
        root = self.open_site_annotations()
        z = root[field][contig]
        d = da.from_array(z, chunks=z.chunks)

        # access and subset to SNP positions
        pos = self.snp_sites(
            contig=contig, field="POS", site_mask=site_mask, site_filters=site_filters
        ).compute()
        d = da.take(d, pos - 1)

        return d

    def snp_dataset(
        self,
        contig,
        sample_sets="v3_wild",
        species_calls=("20200422", "aim"),
        site_mask=None,
        site_filters="dt_20200416",
    ):
        """TODO doc me"""

        # TODO support multiple contigs

        # variant arrays
        pos, ref, alt = self.snp_sites(
            contig=contig, site_mask=site_mask, site_filters=site_filters
        )
        variant_position = pos
        variant_allele = da.concatenate([ref[:, None], alt], axis=1)
        contig_index = self.contigs.index(contig)
        variant_contig = da.full_like(
            variant_position, fill_value=contig_index, dtype="u1"
        )

        # call arrays
        gt = self.snp_genotypes(
            contig=contig,
            sample_sets=sample_sets,
            field="GT",
            site_mask=site_mask,
            site_filters=site_filters,
        )
        call_genotype = gt

        # sample arrays
        df_samples = self.sample_metadata(
            sample_sets=sample_sets, species_calls=species_calls
        )
        sample_id = df_samples["sample_id"]

        # setup data variables
        data_vars = {
            "variant_contig": ([DIM_VARIANT], variant_contig),
            "variant_position": ([DIM_VARIANT], variant_position),
            "variant_allele": ([DIM_VARIANT, DIM_ALLELE], variant_allele),
            "sample_id": ([DIM_SAMPLE], sample_id),
            "call_genotype": ([DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY], call_genotype),
            "call_genotype_mask": (
                [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY],
                call_genotype < 0,
            ),
        }

        # setup attributes
        attrs = {"contigs": self.contigs}

        # create a dataset
        ds = xarray.Dataset(data_vars=data_vars, attrs=attrs)

        return ds

    def open_cnv_hmm(self, sample_set):
        """Open CNV HMM zarr.

        Parameters
        ----------
        sample_set : str

        Returns
        -------
        cnv call root : zarr.hierarchy.Group

        """
        try:
            return self._cache_cnv_hmm[sample_set]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            path = f"{self.path}/{release}/cnv/{sample_set}/hmm/zarr"
            store = SafeStore(self.fs.get_mapper(path))
            root = zarr.open_consolidated(store=store)
            self._cache_cnv_hmm[sample_set] = root
            return root

    def cnv_hmm_load_variants(self, contig, sample_set):

        root = self.open_cnv_hmm(sample_set=sample_set)

        start_z = root[contig]["variants/window_start"]
        window_starts = da.from_array(start_z, start_z.chunks)

        stop_z = root[contig]["variants/window_stop"]
        window_stops = da.from_array(stop_z, stop_z.chunks)

        return window_starts, window_stops

    def cnv_hmm_load_calldata(self, contig, sample_set):

        root = self.open_cnv_hmm(sample_set=sample_set)
        calldata_fields = ("calldata/hmm_state", "calldata/normalized_coverage", "calldata/raw_coverage")

        hmm_z = root[contig]["calldata/hmm_state"]
        hmm = da.from_array(hmm_z, hmm_z.chunks)

        norm_z = root[contig]["calldata/normalized_coverage"]
        norm_cov = da.from_array(norm_z, norm_z.chunks)

        raw_z = root[contig]["calldata/raw_coverage"]
        raw_cov = da.from_array(raw_z, raw_z.chunks)

        return hmm, norm_cov, raw_cov

    def cnv_hmm(self, contig, sample_sets="v3_wild"):

        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)

        if isinstance(sample_sets, str):
            # single sample set
            start_pos, stop_pos = self.cnv_hmm_load_variants(contig=contig, sample_set=sample_sets)

            copy_number, normalised_coverage, raw_coverage = self.cnv_hmm_load_calldata(
                contig=contig, sample_set=sample_sets)

        else:
            # concatenate multiple sample sets using reciprocal function.

            # variants are the same across sample_sets
            start_pos, stop_pos = self.cnv_hmm_load_variants(contig=contig, sample_set=sample_sets[0])

            r = [self.cnv_hmm_load_calldata(contig=contig, sample_set=c) for c in sample_sets]

            # now unpack these
            copy_number = da.concatenate([result[0] for result in r], axis=1)
            normalised_coverage = da.concatenate([result[1] for result in r], axis=1)
            raw_coverage = da.concatenate([result[2] for result in r], axis=1)

        return start_pos, stop_pos, copy_number, raw_coverage, normalised_coverage

    def open_cnv_calls(self, sample_set, analysis):
        """Open CNV calls zarr.

        Parameters
        ----------
        sample_set : str

        Returns
        -------
        cnv call root : zarr.hierarchy.Group

        """
        try:
            return self._cache_cnv_hmm[sample_set]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            path = f"{self.path}/{release}/cnv/{sample_set}/calls/{analysis}/zarr"
            store = SafeStore(self.fs.get_mapper(path))
            root = zarr.open_consolidated(store=store)
            self._cache_cnv_hmm[sample_set] = root
            return root

    def cnv_calls_load_variants(self, contig, sample_set, analysis):

        root = self.open_cnv_calls(sample_set=sample_set, analysis=analysis)

        # these are all variant bits
        start_z = root[contig]["variants/cnv_start"]
        cnv_start = da.from_array(start_z, chunks=start_z.chunks)

        end_z = root[contig]["variants/cnv_end"]
        cnv_end = da.from_array(end_z, chunks=end_z.chunks)

        filter_z = root[contig]["variants/qMerge_FILTER_PASS"]
        cnv_filter = da.from_array(filter_z, chunks=filter_z.chunks)

        return cnv_start, cnv_end, cnv_filter

    def cnv_calls_load_calldata_samples(self, contig, sample_set, analysis):

        root = self.open_cnv_calls(sample_set=sample_set, analysis=analysis)

        gt_z = root[contig]["calldata/GT"]
        gt = da.from_array(gt_z, chunks=gt_z.chunks)

        sample_names = root["samples"][:]

        return sample_names, gt

    def cnv_calls(self, contig, analysis, sample_sets="v3_wild", apply_quantile_filter=False, **kwargs):
        """Load CNV alleles

        Parameters
        ----------

        contig: str
        analysis: str
        sample_sets : str
        apply_quantile_filter: bool
        **kwargs: passed directly to self.sample_metadata eg. species_calls=("20200422", "pca"))

        Returns
        -------
        df_cnv_samples: pd.DataFrame
        nv_start: array
        cnv_stop: array
        cnv_gt: array (n alleles x s samples).

        """

        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)

        df_sample_metadata = self.sample_metadata(sample_sets=sample_sets, **kwargs)

        # if single sample_set...

        if isinstance(sample_sets, str):

            # needs to be concatenated.
            sample_names, genotypes = self.cnv_calls_load_calldata_samples(
                contig=contig, sample_set=sample_sets, analysis=analysis)

            start_pos, end_pos, qmerge_filter = self.cnv_calls_load_variants(
                contig=contig, sample_set=sample_sets, analysis=analysis)

        else:
            # call reciprocal
            # if we are returning multiple sample sets: only want to return one copy of variant data.
            r = [self.cnv_calls_load_calldata_samples(
                    contig=contig, sample_set=c, analysis=analysis) for c in sample_sets]

            # concatenate the above
            sample_names = np.concatenate([result[0] for result in r], axis=0)

            genotypes = da.concatenate([result[1] for result in r], axis=1)

            # this is the sample for all sample_sets
            start_pos, end_pos, qmerge_filter = self.cnv_calls_load_variants(
                contig=contig, sample_set=sample_sets[0], analysis=analysis)

        df_sample_metadata_subset = df_sample_metadata.set_index("sample_id").loc[sample_names].reset_index()

        if apply_quantile_filter:
            genotypes = da.compress(qmerge_filter, genotypes, axis=0)
            start_pos = da.compress(qmerge_filter, start_pos, axis=0)
            end_pos = da.compress(qmerge_filter, end_pos, axis=0)

        return df_sample_metadata_subset, start_pos, end_pos, genotypes

