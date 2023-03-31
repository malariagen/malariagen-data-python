from typing import Tuple

import zarr
from numpydoc_decorator import doc

from ..util import init_zarr_store
from .base import AnophelesBase


class AnophelesGenomeSequenceData(AnophelesBase):
    def __init__(
        self,
        *,
        contigs,
        genome_fasta_path,
        genome_fai_path,
        genome_zarr_path,
        genome_ref_id,
        genome_ref_name,
        **kwargs,
    ):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

        # Store attributes.
        self._contigs = contigs
        self._genome_fasta_path = genome_fasta_path
        self._genome_fai_path = genome_fai_path
        self._genome_zarr_path = genome_zarr_path
        self._genome_ref_id = genome_ref_id
        self._genome_ref_name = genome_ref_name

        # Initialize cache attributes.
        self._cache_genome = None

    @property
    def contigs(self) -> Tuple[str]:
        return self._contigs

    @doc(
        summary="Open the reference genome zarr.",
        returns="Zarr hierarchy containing the reference genome sequence.",
    )
    def open_genome(self) -> zarr.hierarchy.Group:
        if self._cache_genome is None:
            path = f"{self._base_path}/{self._genome_zarr_path}"
            store = init_zarr_store(fs=self._fs, path=path)
            self._cache_genome = zarr.open_consolidated(store=store)
        return self._cache_genome
