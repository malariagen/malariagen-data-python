import pandas as pd  # type: ignore
from pandas import CategoricalDtype
import numpy as np  # type: ignore
import allel  # type: ignore

from numpydoc_decorator import doc
from ..util import check_types, _karyotype_tags_n_alt
from . import base_params
from typing import Optional

from .genome_features import AnophelesGenomeFeaturesData
from .genome_sequence import AnophelesGenomeSequenceData
from .sample_metadata import AnophelesSampleMetadata
from .karyotype_params import inversion_param


class AnophelesKaryotypeData(
    AnophelesSampleMetadata, AnophelesGenomeFeaturesData, AnophelesGenomeSequenceData
):
    def __init__(
        self,
        **kwargs,
    ):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

    @check_types
    @doc(
        summary="Load tag SNPs for a given inversion in Ag.",
    )
    def load_inversion_tags(self, inversion: inversion_param) -> pd.DataFrame:
        # needs to be modified depending on where we are hosting
        import importlib.resources
        from .. import resources

        with importlib.resources.path(resources, "karyotype_tag_snps.csv") as path:
            df_tag_snps = pd.read_csv(path, sep=",")
        return df_tag_snps.query(f"inversion == '{inversion}'").reset_index()

    @check_types
    @doc(
        summary="Infer karyotype from tag SNPs for a given inversion in Ag.",
    )
    def karyotype(
        self,
        inversion: inversion_param,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
    ) -> pd.DataFrame:
        # load tag snp data
        df_tagsnps = self.load_inversion_tags(inversion=inversion)
        inversion_pos = df_tagsnps["position"]
        inversion_alts = df_tagsnps["alt_allele"]
        contig = inversion[0:2]

        # get snp calls for inversion region
        start, end = np.min(inversion_pos), np.max(inversion_pos)
        region = f"{contig}:{start}-{end}"

        ds_snps = self.snp_calls(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
        )

        with self._spinner("Inferring karyotype from tag SNPs"):
            # access variables we need
            geno = allel.GenotypeDaskArray(ds_snps["call_genotype"].data)
            pos = allel.SortedIndex(ds_snps["variant_position"].values)
            samples = ds_snps["sample_id"].values
            alts = ds_snps["variant_allele"].values.astype(str)

            # subset to position of inversion tags
            mask = pos.locate_intersection(inversion_pos)[0]
            alts = alts[mask]
            geno = geno.compress(mask, axis=0).compute()

            # infer karyotype
            gn_alt = _karyotype_tags_n_alt(
                gt=geno, alts=alts, inversion_alts=inversion_alts
            )
            is_called = geno.is_called()

            # calculate mean genotype for each sample whilst masking missing calls
            av_gts = np.mean(np.ma.MaskedArray(gn_alt, mask=~is_called), axis=0)
            total_sites = np.sum(is_called, axis=0)

            df = pd.DataFrame(
                {
                    "sample_id": samples,
                    "inversion": inversion,
                    f"karyotype_{inversion}_mean": av_gts,
                    # round the genotypes then convert to int
                    f"karyotype_{inversion}": av_gts.round().astype(int),
                    "total_tag_snps": total_sites,
                },
            )
            # Allow filling missing values with "<NA>" visible placeholder.
            kt_dtype = CategoricalDtype(categories=[0, 1, 2, "<NA>"], ordered=True)
            df[f"karyotype_{inversion}"] = df[f"karyotype_{inversion}"].astype(kt_dtype)

        return df
