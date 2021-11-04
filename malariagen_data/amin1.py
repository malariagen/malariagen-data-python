import pandas as pd
import zarr

from .util import (
    da_from_zarr,
    init_filesystem,
    init_zarr_store,
    read_gff3,
    unpack_gff3_attributes,
)

geneset_gff3_path = (
    "reference/genome/aminm1/Anopheles-minimus-MINIMUS1_BASEFEATURES_AminM1.8.gff3.gz"
)
genome_zarr_path = "reference/genome/aminm1/VectorBase-48_AminimusMINIMUS1_Genome.zarr"


class Amin1:
    def __init__(self, url, **kwargs):

        # setup filesystem
        self._fs, self._path = init_filesystem(url, **kwargs)

        # setup caches
        self._cache_sample_metadata = None
        self._cache_genome = None
        self._cache_geneset = dict()
        self._contigs = None

    @property
    def contigs(self):
        if self._contigs is None:
            self._contigs = tuple(sorted(self.open_genome()))
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
            df = self._cache_geneset[attributes]

        except KeyError:
            path = f"{self._path}/{geneset_gff3_path}"
            with self._fs.open(path, mode="rb") as f:
                df = read_gff3(f, compression="gzip")
            if attributes is not None:
                df = unpack_gff3_attributes(df, attributes=attributes)
            self._cache_geneset[attributes] = df

        return df

    def snp_calls(self, contig, site_mask=False):
        pass
