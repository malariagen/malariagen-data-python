import dask.array as da
import pandas as pd
import xarray
import zarr

from .util import (
    DIM_ALLELE,
    DIM_PLOIDY,
    DIM_SAMPLE,
    DIM_VARIANT,
    Region,
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

GENOME_FEATURES_GFF3_PATH = (
    "reference/genome/aminm1/Anopheles-minimus-MINIMUS1_BASEFEATURES_AminM1.8.gff3.gz"
)
genome_zarr_path = "reference/genome/aminm1/VectorBase-48_AminimusMINIMUS1_Genome.zarr"


DEFAULT_URL = "gs://vo_amin_release/"


class Amin1:
    def __init__(self, url=DEFAULT_URL, **kwargs):

        # setup filesystem
        self._fs, self._path = init_filesystem(url, **kwargs)

        # setup caches
        self._cache_sample_metadata = None
        self._cache_genome = None
        self._cache_genome_features = dict()
        self._cache_snp_genotypes = None
        self._contigs = None

    @property
    def contigs(self):
        if self._contigs is None:
            # only include the contigs that were genotyped - 40 largest
            self._contigs = tuple(
                k for k in sorted(self.open_snp_calls()) if k.startswith("KB")
            )
        return self._contigs

    def sample_metadata(self):
        """Access sample metadata.

        Returns
        -------
        df : pandas.DataFrame

        """
        if self._cache_sample_metadata is None:
            path = f"{self._path}/v1/metadata/samples.meta.csv"
            with self._fs.open(path) as f:
                self._cache_sample_metadata = pd.read_csv(f, na_values="")
        return self._cache_sample_metadata

    def open_genome(self):
        """Open the reference genome zarr.

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        if self._cache_genome is None:
            path = f"{self._path}/{genome_zarr_path}"
            store = init_zarr_store(fs=self._fs, path=path)
            self._cache_genome = zarr.open_consolidated(store=store)
        return self._cache_genome

    def genome_sequence(self, region, inline_array=True, chunks="native"):
        """Access the reference genome sequence.

        Parameters
        ----------
        region: str or list of str or Region
            Contig (e.g., "KB663610"), gene name (e.g., "AMIN002150"), genomic region
            defined with coordinates (e.g., "KB663610:1-100000") or a named tuple with
            genomic location `Region(contig, start, end)`. Multiple values can be provided
            as a list, in which case data will be concatenated.
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

    def geneset(self, *args, **kwargs):
        """Deprecated, this method has been renamed to genome_features()."""
        return self.genome_features(*args, **kwargs)

    def genome_features(self, attributes=("ID", "Parent", "Name", "description")):
        """Access genome feature annotations.

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
            df = self._cache_genome_features[attributes]

        except KeyError:
            path = f"{self._path}/{GENOME_FEATURES_GFF3_PATH}"
            with self._fs.open(path, mode="rb") as f:
                df = read_gff3(f, compression="gzip")
            if attributes is not None:
                df = unpack_gff3_attributes(df, attributes=attributes)
            self._cache_genome_features[attributes] = df

        return df

    def open_snp_calls(self):
        if self._cache_snp_genotypes is None:
            path = f"{self._path}/v1/snp_genotypes/all"
            store = init_zarr_store(fs=self._fs, path=path)
            self._cache_snp_genotypes = zarr.open_consolidated(store=store)
        return self._cache_snp_genotypes

    def _snp_calls_dataset(self, *, region, inline_array, chunks):

        assert isinstance(region, Region)
        contig = region.contig

        # setup
        coords = dict()
        data_vars = dict()
        root = self.open_snp_calls()

        # variant_position
        pos_z = root[f"{contig}/variants/POS"]
        variant_position = da_from_zarr(pos_z, inline_array=inline_array, chunks=chunks)
        coords["variant_position"] = [DIM_VARIANT], variant_position

        # variant_allele
        ref_z = root[f"{contig}/variants/REF"]
        alt_z = root[f"{contig}/variants/ALT"]
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

        # variant_filter_pass
        fp_z = root[f"{contig}/variants/filter_pass"]
        fp = da_from_zarr(fp_z, inline_array=inline_array, chunks=chunks)
        data_vars["variant_filter_pass"] = [DIM_VARIANT], fp

        # call arrays
        gt_z = root[f"{contig}/calldata/GT"]
        call_genotype = da_from_zarr(gt_z, inline_array=inline_array, chunks=chunks)
        gq_z = root[f"{contig}/calldata/GQ"]
        call_gq = da_from_zarr(gq_z, inline_array=inline_array, chunks=chunks)
        ad_z = root[f"{contig}/calldata/AD"]
        call_ad = da_from_zarr(ad_z, inline_array=inline_array, chunks=chunks)
        mq_z = root[f"{contig}/calldata/MQ"]
        call_mq = da_from_zarr(mq_z, inline_array=inline_array, chunks=chunks)
        data_vars["call_genotype"] = (
            [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY],
            call_genotype,
        )
        data_vars["call_GQ"] = ([DIM_VARIANT, DIM_SAMPLE], call_gq)
        data_vars["call_MQ"] = ([DIM_VARIANT, DIM_SAMPLE], call_mq)
        data_vars["call_AD"] = ([DIM_VARIANT, DIM_SAMPLE, DIM_ALLELE], call_ad)

        # sample arrays
        z = root["samples"]
        sample_id = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        coords["sample_id"] = [DIM_SAMPLE], sample_id

        # setup attributes
        attrs = {"contigs": self.contigs}

        # create a dataset
        ds = xarray.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        # deal with region
        if region.start or region.end:
            loc_region = locate_region(region, pos_z)
            ds = ds.isel(variants=loc_region)

        return ds

    def snp_calls(self, region, site_mask=False, inline_array=True, chunks="native"):
        """Access SNP sites, site filters and genotype calls.

        Parameters
        ----------
        region: str or list of str or Region
            Contig (e.g., "KB663610"), gene name (e.g., "AMIN002150"), genomic region
            defined with coordinates (e.g., "KB663610:1-100000") or a named tuple with
            genomic location `Region(contig, start, end)`. Multiple values can be provided
            as a list, in which case data will be concatenated.
        site_mask : bool
            Apply site filters.
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr chunks.
            Also can be a target size, e.g., '200 MiB'.

        Returns
        -------
        ds : xarray.Dataset

        """

        region = resolve_region(self, region)

        # normalise to simplify concatenation logic
        if isinstance(region, str) or isinstance(region, Region):
            region = [region]

        # concatenate along variants dimension
        datasets = [
            self._snp_calls_dataset(
                region=r,
                inline_array=inline_array,
                chunks=chunks,
            )
            for r in region
        ]
        ds = xarray_concat(
            datasets,
            dim=DIM_VARIANT,
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="override",
        )

        # apply site filters
        if site_mask:
            ds = dask_compress_dataset(
                ds, indexer="variant_filter_pass", dim=DIM_VARIANT
            )

            # add call_genotype_mask
        ds["call_genotype_mask"] = ds["call_genotype"] < 0

        return ds
