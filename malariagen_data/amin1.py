import zarr
from fsspec.mapping import FSMap

from .util import SafeStore, from_zarr, init_filesystem


class Amin1:
    def __init__(self, url, **kwargs):

        # setup filesystem
        self._fs, self._path = init_filesystem(url, **kwargs)

        # setup caches
        self._cache_sample_metadata = dict()
        self._cache_genome = None
        self._cache_geneset = dict()

    @property
    def cohorts(self):
        return None

    def sample_metadata(self):
        pass

    def open_genome(self):
        """Open the reference genome zarr.

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        if self._cache_genome is None:
            path = f"{self._path}/reference/genome/aminm1/VectorBase-48_AminimusMINIMUS1_Genome.zarr"
            store = SafeStore(FSMap(root=path, fs=self._fs, check=False, create=False))
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
        d = from_zarr(z, inline_array=inline_array, chunks=chunks)
        return d

    def geneset(self, attributes=None):
        pass

    def snp_calls(self, contig, site_mask=False):
        pass
