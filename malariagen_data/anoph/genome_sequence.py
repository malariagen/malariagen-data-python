from typing import Tuple

import zarr
from numpydoc_decorator import doc

from ..util import init_zarr_store
from .base import AnophelesBase


class AnophelesGenomeSequenceData(AnophelesBase):
    def __init__(self, **kwargs):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor, and the superclass constructor
        # is called first.
        super().__init__(**kwargs)

        # Initialize cache attributes.
        self._cache_genome = None

    @property
    def contigs(self) -> Tuple[str, ...]:
        return tuple(self.config["CONTIGS"])

    @property
    def _genome_zarr_path(self) -> str:
        return self.config["GENOME_ZARR_PATH"]

    @property
    def _genome_fasta_path(self) -> str:
        return self.config["GENOME_FASTA_PATH"]

    @property
    def _genome_fai_path(self) -> str:
        return self.config["GENOME_FAI_PATH"]

    @property
    def _genome_ref_id(self) -> str:
        return self.config["GENOME_REF_ID"]

    @property
    def _genome_ref_name(self) -> str:
        return self.config["GENOME_REF_NAME"]

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
