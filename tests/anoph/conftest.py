import json
import shutil
import string
from pathlib import Path
from random import choice, choices, randint
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr  # type: ignore
from malariagen_data.util import _gff3_parse_attributes

# N.B., this file (conftest.py) is handled in a special way
# by pytest. In short, this file is a place to define any
# fixtures which are needed across multiple test modules
# within the current directory. For more information see the
# following links:
#
# https://docs.pytest.org/en/7.2.x/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
# https://stackoverflow.com/questions/34466027/in-pytest-what-is-the-use-of-conftest-py-files
#
# Note that any fixtures defined here are automatically available
# in test modules, they do not need to be imported.
#
# In the "Fixture" classes below we are going to create some
# data locally which follows the same layout and format of the
# real data in GCS, but which is much smaller and so can be used
# for faster test runs.


@pytest.fixture(scope="session")
def fixture_dir():
    cwd = Path(__file__).parent.resolve()
    return cwd / "fixture"


def simulate_contig(*, low, high, base_composition):
    size = np.random.randint(low=low, high=high)
    bases = np.array([b"a", b"c", b"g", b"t", b"n", b"A", b"C", b"G", b"T", b"N"])
    p = np.array([base_composition[b] for b in bases])
    seq = np.random.choice(bases, size=size, replace=True, p=p)
    return seq


def simulate_genome(*, path, contigs, low, high, base_composition):
    path.mkdir(parents=True, exist_ok=True)
    root = zarr.open(path, mode="w")
    for contig in contigs:
        seq = simulate_contig(low=low, high=high, base_composition=base_composition)
        root.create_dataset(name=contig, data=seq)
    zarr.consolidate_metadata(path)
    return root


class Gff3Simulator:
    def __init__(
        self,
        *,
        contig_sizes,
        gene_type="gene",
        transcript_type="mRNA",
        exon_type="exon",
        utr5_type="five_prime_UTR",
        utr3_type="three_prime_UTR",
        cds_type="CDS",
        inter_size_low=1_000,
        inter_size_high=10_000,
        gene_size_low=300,
        gene_size_high=30_000,
        n_transcripts_low=1,
        n_transcripts_high=3,
        n_exons_low=1,
        n_exons_high=5,
        intron_size_low=10,
        intron_size_high=100,
        exon_size_low=100,
        exon_size_high=1_000,
        source="random",
        max_genes=1_000,
        attrs=("Name", "description"),
    ):
        self.contig_sizes = contig_sizes
        self.gene_type = gene_type
        self.transcript_type = transcript_type
        self.exon_type = exon_type
        self.utr5_type = utr5_type
        self.utr3_type = utr3_type
        self.cds_type = cds_type
        self.inter_size_low = inter_size_low
        self.inter_size_high = inter_size_high
        self.gene_size_low = gene_size_low
        self.gene_size_high = gene_size_high
        self.n_transcripts_low = n_transcripts_low
        self.n_transcripts_high = n_transcripts_high
        self.n_exons_low = n_exons_low
        self.n_exons_high = n_exons_high
        self.intron_size_low = intron_size_low
        self.intron_size_high = intron_size_high
        self.exon_size_low = exon_size_low
        self.exon_size_high = exon_size_high
        self.source = source
        self.max_genes = max_genes
        self.attrs = attrs

    def simulate_gff(self, *, path):
        dfs = []
        for contig, contig_size in self.contig_sizes.items():
            df = self.simulate_contig(contig=contig, contig_size=contig_size)
            dfs.append(df)
        df_gf = pd.concat(dfs, axis=0, ignore_index=True)
        df_gf.to_csv(
            path,
            sep="\t",
            header=False,
            index=False,
        )
        return df_gf

    def simulate_contig(self, *, contig, contig_size):
        sim = self.simulate_genes(
            contig=contig,
            contig_size=contig_size,
        )
        df = pd.DataFrame(
            sim,
            columns=[
                "seqid",
                "source",
                "type",
                "start",
                "end",
                "score",
                "strand",
                "phase",
                "attributes",
            ],
        )
        return df

    def simulate_genes(self, *, contig, contig_size):
        # Keep track of where we are on the contig. This allows for overlapping
        # features on opposite strands.
        cur_fwd = 1
        cur_rev = 1

        # Simulate genes.
        for gene_ix in range(self.max_genes):
            gene_id = f"gene-{contig}-{gene_ix}"
            strand = choice(["+", "-"])
            inter_size = randint(self.inter_size_low, self.inter_size_high)
            gene_size = randint(self.gene_size_low, self.gene_size_high)
            if strand == "+":
                gene_start = cur_fwd + inter_size
            else:
                gene_start = cur_rev + inter_size
            gene_end = gene_start + gene_size
            if gene_end >= contig_size:
                # Bail out, no more space left on the contig.
                return
            assert gene_end > gene_start
            gene_attrs = f"ID={gene_id}"
            for attr in self.attrs:
                random_str = "".join(
                    choices(string.ascii_uppercase + string.digits, k=5)
                )
                gene_attrs += f";{attr}={random_str}"
            gene = (
                contig,
                self.source,
                self.gene_type,
                gene_start,
                gene_end,
                ".",
                strand,
                ".",
                gene_attrs,
            )
            yield gene

            yield from self.simulate_transcripts(
                contig=contig,
                strand=strand,
                gene_ix=gene_ix,
                gene_id=gene_id,
                gene_start=gene_start,
                gene_end=gene_end,
            )

            # Update state.
            if strand == "+":
                cur_fwd = gene_end
            else:
                cur_rev = gene_end

    def simulate_transcripts(
        self,
        *,
        contig,
        strand,
        gene_ix,
        gene_id,
        gene_start,
        gene_end,
    ):
        # Note that this always models transcripts as starting and ending at
        # the same coordinates as the parent gene, which is not strictly
        # accurate in real data.

        for transcript_ix in range(
            randint(self.n_transcripts_low, self.n_transcripts_high)
        ):
            transcript_id = f"transcript-{contig}-{gene_ix}-{transcript_ix}"
            transcript_start = gene_start
            transcript_end = gene_end
            assert transcript_end > transcript_start
            transcript = (
                contig,
                self.source,
                self.transcript_type,
                transcript_start,
                transcript_end,
                ".",
                strand,
                ".",
                f"ID={transcript_id};Parent={gene_id}",
            )
            yield transcript

            yield from self.simulate_exons(
                contig=contig,
                strand=strand,
                gene_ix=gene_ix,
                transcript_ix=transcript_ix,
                transcript_id=transcript_id,
                transcript_start=transcript_start,
                transcript_end=transcript_end,
            )

    def simulate_exons(
        self,
        *,
        contig,
        strand,
        gene_ix,
        transcript_ix,
        transcript_id,
        transcript_start,
        transcript_end,
    ):
        # Note that this doesn't correctly model the style of GFF for
        # funestus, because here each exon has a single parent transcript,
        # whereas in the funestus annotations, exons can be shared between
        # transcripts.

        transcript_size = transcript_end - transcript_start
        exons = []
        exon_end = transcript_start
        n_exons = randint(self.n_exons_low, self.n_exons_high)
        for exon_ix in range(n_exons):
            exon_id = f"exon-{contig}-{gene_ix}-{transcript_ix}-{exon_ix}"
            if exon_ix > 0:
                # Insert an intron between this exon and the previous one.
                intron_size = randint(
                    self.intron_size_low, min(transcript_size, self.intron_size_high)
                )
                exon_start = exon_end + intron_size
                if exon_start >= transcript_end:
                    # Stop making exons, no more space left in the transcript.
                    break
            else:
                # First exon, assume exon starts where the transcript starts.
                exon_start = transcript_start
            exon_size = randint(self.exon_size_low, self.exon_size_high)
            exon_end = min(exon_start + exon_size, transcript_end)
            assert exon_end > exon_start
            exon = (
                contig,
                self.source,
                self.exon_type,
                exon_start,
                exon_end,
                ".",
                strand,
                ".",
                f"ID={exon_id};Parent={transcript_id}",
            )
            yield exon
            exons.append(exon)

        # Note that this is not perfect, because in reality an exon can contain
        # part of a UTR and part of a CDS, but that is harder to simulate. So
        # keep things simple for now.
        if strand == "-":
            # Take exons in reverse order.
            exons == exons[::-1]
        for exon_ix, exon in enumerate(exons):
            first_exon = exon_ix == 0
            last_exon = exon_ix == len(exons) - 1
            # Ensure at least one CDS.
            if first_exon and len(exons) > 1:
                feature_type = self.utr5_type
                phase = "."
            elif last_exon and len(exons) > 2:
                feature_type = self.utr3_type
                phase = "."
            else:
                feature_type = self.cds_type
                # Cheat a little, random phase.
                phase = choice([1, 2, 3])
            feature = (
                contig,
                self.source,
                feature_type,
                exon[3],
                exon[4],
                ".",
                strand,
                phase,
                f"Parent={transcript_id}",
            )
            yield feature


def simulate_snp_sites(path, contigs, genome):
    root = zarr.open(path, mode="w")
    n_sites = dict()

    for contig in contigs:
        # Obtain variants group.
        variants = root.require_group(contig).require_group("variants")

        # Simulate POS.
        seq = genome[contig][:]
        loc_n = (seq == b"N") | (seq == b"n")
        pos = np.nonzero(~loc_n)[0] + 1  # 1-based coordinates
        variants.create_dataset(name="POS", data=pos.astype("i4"))

        # Simulate REF.
        ref = np.char.upper(seq[~loc_n])  # ensure upper case
        assert pos.shape == ref.shape
        variants.create_dataset(name="REF", data=ref)

        # Simulate ALT.
        alt = np.empty(shape=(ref.shape[0], 3), dtype="S1")
        alt[ref == b"A"] = np.array([b"C", b"T", b"G"])
        alt[ref == b"C"] = np.array([b"A", b"T", b"G"])
        alt[ref == b"T"] = np.array([b"A", b"C", b"G"])
        alt[ref == b"G"] = np.array([b"A", b"C", b"T"])
        variants.create_dataset(name="ALT", data=alt)

        # Store number of sites for later.
        n_sites[contig] = pos.shape[0]

    zarr.consolidate_metadata(path)
    return root, n_sites


def simulate_site_filters(path, contigs, p_pass, n_sites):
    root = zarr.open(path, mode="w")
    p = np.array([1 - p_pass, p_pass])
    for contig in contigs:
        variants = root.require_group(contig).require_group("variants")
        size = n_sites[contig]
        filter_pass = np.random.choice([False, True], size=size, p=p)
        variants.create_dataset(name="filter_pass", data=filter_pass)
    zarr.consolidate_metadata(path)


def simulate_snp_genotypes(
    zarr_path, metadata_path, contigs, n_sites, p_allele, p_missing
):
    root = zarr.open(zarr_path, mode="w")

    # Create samples array.
    df_samples = pd.read_csv(metadata_path, engine="python")
    n_samples = len(df_samples)
    samples = df_samples["sample_id"].values.astype("S")
    root.create_dataset(name="samples", data=samples)

    for contig in contigs:
        # Set up groups.
        contig_grp = root.require_group(contig)
        calldata = contig_grp.require_group("calldata")
        contig_n_sites = n_sites[contig]

        # Simulate genotype calls.
        gt = np.random.choice(
            np.arange(4, dtype="i1"),
            size=(contig_n_sites, n_samples, 2),
            replace=True,
            p=p_allele,
        )

        # Simulate missing calls.
        n_calls = contig_n_sites * n_samples
        loc_missing = np.random.choice(
            [False, True], size=n_calls, replace=True, p=p_missing
        )
        gt.reshape(-1, 2)[loc_missing] = -1

        # Store genotype calls.
        # N.B., we need to chunk across the final dimension here,
        # otherwise allele count computation breaks inside scikit-allel.
        gt_chunks = (contig_n_sites // 5, n_samples // 3, None)
        calldata.create_dataset(name="GT", data=gt, chunks=gt_chunks)

        # Create other arrays - these are never actually used currently
        # so we'll create some empty arrays to avoid delaying the tests.
        calldata.create_dataset(
            name="GQ", shape=(contig_n_sites, n_samples), dtype="i1", fill_value=-1
        )
        calldata.create_dataset(
            name="MQ", shape=(contig_n_sites, n_samples), dtype="f4", fill_value=-1
        )
        calldata.create_dataset(
            name="AD",
            shape=(contig_n_sites, n_samples, 4),
            dtype="i2",
            fill_value=-1,
        )

    zarr.consolidate_metadata(zarr_path)


def simulate_site_annotations(path, genome):
    root = zarr.open(path, mode="w")
    contigs = list(genome)

    # Take a very simple approach here to simulate random data.
    # It won't be biologically realistic, but should hopefully
    # suffice for testing purposes.

    # codon_degeneracy
    grp = root.require_group("codon_degeneracy")
    vals = np.arange(-1, 5)
    p = [0.897754, 0.0, 0.060577, 0.014287, 0.011096, 0.016286]
    for contig in contigs:
        size = genome[contig].shape[0]
        x = np.random.choice(vals, size=size, replace=True, p=p)
        grp.create_dataset(name=contig, data=x)

    # codon_nonsyn
    grp = root.require_group("codon_nonsyn")
    vals = np.arange(4)
    p = [0.91404, 0.001646, 0.018698, 0.065616]
    for contig in contigs:
        size = genome[contig].shape[0]
        x = np.random.choice(vals, size=size, replace=True, p=p)
        grp.create_dataset(name=contig, data=x)

    # codon_position
    grp = root.require_group("codon_position")
    vals = np.arange(4)
    p = [0.897754, 0.034082, 0.034082, 0.034082]
    for contig in contigs:
        size = genome[contig].shape[0]
        x = np.random.choice(vals, size=size, replace=True, p=p)
        grp.create_dataset(name=contig, data=x)

    # seq_cls
    grp = root.require_group("seq_cls")
    vals = np.arange(11)
    p = [
        0.034824,
        0.230856,
        0.318803,
        0.009675,
        0.015201,
        0.015446,
        0.059981,
        0.018995,
        0.085244,
        0.180545,
        0.03043,
    ]
    for contig in contigs:
        size = genome[contig].shape[0]
        x = np.random.choice(vals, size=size, replace=True, p=p)
        grp.create_dataset(name=contig, data=x)

    # seq_flen
    grp = root.require_group("seq_flen")
    for contig in contigs:
        size = genome[contig].shape[0]
        x = np.random.randint(low=0, high=40_000, size=size)
        grp.create_dataset(name=contig, data=x)

    # seq_relpos_start
    grp = root.require_group("seq_relpos_start")
    for contig in contigs:
        size = genome[contig].shape[0]
        x = np.random.beta(a=0.4, b=4, size=size) * 40_000
        grp.create_dataset(name=contig, data=x)

    # seq_relpos_stop
    grp = root.require_group("seq_relpos_stop")
    for contig in contigs:
        size = genome[contig].shape[0]
        x = np.random.beta(a=0.4, b=4, size=size) * 40_000
        grp.create_dataset(name=contig, data=x)

    zarr.consolidate_metadata(path)


def simulate_hap_sites(path, contigs, snp_sites, p_site):
    n_sites = dict()
    root = zarr.open(path, mode="w")

    for contig in contigs:
        # Obtain variants group.
        variants = root.require_group(contig).require_group("variants")

        # Simulate POS.
        snp_pos = snp_sites[f"{contig}/variants/POS"][:]
        loc_hap_sites = np.random.choice(
            [False, True], size=snp_pos.shape[0], p=[1 - p_site, p_site]
        )
        pos = snp_pos[loc_hap_sites]
        variants.create_dataset(name="POS", data=pos)

        # Simulate REF.
        snp_ref = snp_sites[f"{contig}/variants/REF"][:]
        ref = snp_ref[loc_hap_sites]
        variants.create_dataset(name="REF", data=ref)

        # Simulate ALT.
        snp_alt = snp_sites[f"{contig}/variants/ALT"][:]
        sim_alt_choice = np.random.choice(3, size=pos.shape[0])
        alt = np.take_along_axis(
            snp_alt[loc_hap_sites], indices=sim_alt_choice[:, None], axis=1
        )[:, 0]
        variants.create_dataset(name="ALT", data=alt)

        n_sites[contig] = pos.shape[0]

    zarr.consolidate_metadata(path)
    return root, n_sites


def simulate_aim_variants(path, contigs, snp_sites, n_sites_low, n_sites_high):
    ds = xr.Dataset()
    variant_position_arrays = []
    variant_allele_arrays = []
    variant_contig_arrays = []
    for contig_index, contig in enumerate(contigs):
        # Simulate AIM positions variable.
        snp_pos = snp_sites[f"{contig}/variants/POS"][:]
        loc_aim_sites = np.random.choice(
            snp_pos.shape[0], size=np.random.randint(n_sites_low, n_sites_high)
        )
        loc_aim_sites.sort()
        aim_pos = snp_pos[loc_aim_sites]
        variant_position_arrays.append(aim_pos)

        # Simulate contig variable.
        aim_contig = np.full(shape=aim_pos.shape, fill_value=contig_index, dtype="u1")
        variant_contig_arrays.append(aim_contig)

        # Simulate AIM alleles variable.
        snp_ref = snp_sites[f"{contig}/variants/REF"][:]
        snp_alt = snp_sites[f"{contig}/variants/ALT"][:]
        snp_alleles = np.concatenate([snp_ref[:, None], snp_alt], axis=1)
        aim_site_snp_alleles = snp_alleles[loc_aim_sites]
        sim_allele_choice = np.vstack(
            [
                np.random.choice(4, size=2, replace=False)
                for _ in range(len(loc_aim_sites))
            ]
        )
        aim_alleles = np.take_along_axis(
            aim_site_snp_alleles, indices=sim_allele_choice, axis=1
        )
        variant_allele_arrays.append(aim_alleles)

    # Concatenate over contigs.
    variant_contig = np.concatenate(variant_contig_arrays, axis=0)
    ds["variant_contig"] = ("variants",), variant_contig
    variant_position = np.concatenate(variant_position_arrays, axis=0)
    ds["variant_position"] = ("variants",), variant_position
    variant_allele = np.concatenate(variant_allele_arrays, axis=0)
    ds["variant_allele"] = ("variants", "alleles"), variant_allele

    # Dataset attributes.
    ds.attrs["contigs"] = contigs

    # Write dataset to zarr.
    ds.to_zarr(path, mode="w", consolidated=True)

    return ds


def simulate_cnv_hmm(zarr_path, metadata_path, contigs, contig_sizes):
    # zarr_path is the output path to the zarr store
    # metadata_path is the input path for the sample metadata
    # contigs is the list of contigs, e.g. Ag has ('2L', '2R', '3R', '3L', 'X')
    # contig_sizes is a dictionary of the sizes of the contigs in base pairs

    # {release}/cnv/{sample_set}/hmm/zarr
    # - {contig}
    #   - calldata
    #     - CN [2D array] [int] [-1 to 12 for n_variants for n_samples]
    #     - NormCov [2D array] [float] [0 to 356+ for n_variants for for n_samples]
    #     - RawCov [2D array] [int] [-1 to 18465+ for n_variants for for n_samples]
    #   - samples [1D array] [str for n_samples]
    #   - variants
    #      - END [1D array] [int for n_variants]
    #      - POS [1D array] [int for n_variants]
    # - sample_coverage_variance [1D array] [float] [0 to 0.5 for n_samples]
    # - sample_is_high_variance [1D array] [bool] [True or False for n_samples]
    # - samples [1D array] [str]

    # Get a random probability for a sample being high variance, between 0 and 1.
    p_variance = np.random.random()

    # Open a zarr at the specified path.
    root = zarr.open(zarr_path, mode="w")

    # Create samples array.
    df_samples = pd.read_csv(metadata_path, engine="python")
    samples = df_samples["sample_id"].values
    root.create_dataset(name="samples", data=samples, dtype=str)

    # Get the number of samples.
    n_samples = len(df_samples)

    # Simulate sample_coverage_variance array.
    sample_coverage_variance = np.random.uniform(low=0, high=0.5, size=n_samples)
    root.create_dataset(name="sample_coverage_variance", data=sample_coverage_variance)

    # Simulate sample_is_high_variance array.
    sample_is_high_variance = np.random.choice(
        [False, True], size=n_samples, p=[1 - p_variance, p_variance]
    )
    root.create_dataset(name="sample_is_high_variance", data=sample_is_high_variance)

    for contig in contigs:
        # Create the contig group.
        contig_grp = root.require_group(contig)

        # Create the calldata group for this contig.
        calldata_grp = contig_grp.require_group("calldata")

        # Get the length of this contig.
        contig_length_bp = contig_sizes[contig]

        # Set the window size.
        window_size_bp = 300

        # Get the number of non-overlapping windows ("variants") using contig_length.
        # Note: this uses the floor division operator `//`, which returns an integer.
        n_windows = contig_length_bp // window_size_bp

        # Produce the set of window start positions as a tuple (immutable list).
        window_start_pos = tuple(1 + i * window_size_bp for i in range(n_windows))

        # Produce the set of window end positions as a tuple (immutable list).
        window_end_pos = tuple(i * window_size_bp for i in range(1, n_windows)) + (
            contig_length_bp,
        )

        # Simulate CN, NormCov, RawCov under calldata.
        cn = np.random.randint(low=-1, high=12, size=(n_windows, n_samples))
        normCov = np.random.randint(low=0, high=356, size=(n_windows, n_samples))
        rawCov = np.random.randint(low=-1, high=18465, size=(n_windows, n_samples))
        calldata_grp.create_dataset(name="CN", data=cn)
        calldata_grp.create_dataset(name="NormCov", data=normCov)
        calldata_grp.create_dataset(name="RawCov", data=rawCov)

        # Create the samples dataset (again) for this contig.
        contig_grp.create_dataset(name="samples", data=samples, dtype=str)

        # Create variants group for this contig.
        variants_grp = contig_grp.require_group("variants")

        # Simulate POS under variants.
        variants_grp.create_dataset(name="POS", data=window_start_pos)

        # Simulate END under variants.
        variants_grp.create_dataset(name="END", data=window_end_pos)

    zarr.consolidate_metadata(zarr_path)


def simulate_cnv_coverage_calls(zarr_path, metadata_path, contigs, contig_sizes):
    # zarr_path is the output path to the zarr store
    # metadata_path is the input path for the sample metadata
    # contigs is the list of contigs, e.g. Ag has ('2L', '2R', '3R', '3L', 'X')
    # contig_sizes is a dictionary of the sizes of the contigs in base pairs

    # {release}/cnv/{sample_set}/coverage_calls/{analysis}/zarr
    #   - samples [1D array] [str for n_samples]
    #   - {contig}
    #      - calldata
    #         - GT [2D array] [int] [0 or 1 for n_variants for n_samples]
    #      - samples [1D array] [str for n_samples]
    #      - variants
    #         - CIEND [1D array] [int] [0 to 13200+ for n_variants]
    #         - CIPOS [1D array] [int] [0 to 37200+ for n_variants]
    #         - END [1D array] [int for n_variants]
    #         - FILTER_PASS [1D array] [bool] [True or False for n_variants]
    #         - FILTER_qMerge [1D array] [bool] [True or False for n_variants]
    #         - ID [1D array] [unique str for n_variants]
    #         - POS [1D array] [int for n_variants]

    # Get a random probability for choosing allele 1, between 0 and 1.
    p_allele = np.random.random()

    # Get a random probability for passing a particular SNP site (position), between 0 and 1.
    p_filter_pass = np.random.random()

    # Get a random probability for applying qMerge filter to a particular SNP site (position), between 0 and 1.
    p_filter_qMerge = np.random.random()

    # Open a zarr at the specified path.
    root = zarr.open(zarr_path, mode="w")

    # Create samples array.
    df_samples = pd.read_csv(metadata_path, engine="python")
    n_samples = len(df_samples)
    samples = df_samples["sample_id"].values
    root.create_dataset(name="samples", data=samples, dtype=str)

    for contig in contigs:
        # Create the contig group.
        contig_grp = root.require_group(contig)

        # Create the calldata group for this contig.
        calldata = contig_grp.require_group("calldata")

        # Get the length of this contig
        contig_length_bp = contig_sizes[contig]

        # Get a random number of CNV alleles ("variants") to simulate.
        n_cnv_alleles = np.random.randint(1, 5_000)

        # Produce a set of random start positions for each allele as a sorted list.
        allele_start_pos = sorted(
            np.random.randint(1, contig_length_bp, size=n_cnv_alleles)
        )

        # Produce a set of random allele lengths for each allele, according to a range.
        allele_length_bp_min = 100
        allele_length_bp_max = 100_000
        allele_lengths_bp = np.random.randint(
            allele_length_bp_min, allele_length_bp_max, size=n_cnv_alleles
        )

        # Produce the set of end postions for each allele, according to start position and length.
        allele_end_pos = [
            start_pos + length
            for start_pos, length in zip(allele_start_pos, allele_lengths_bp)
        ]

        # Simulate the genotype calls.
        # Note: this is only 2D, unlike SNP, HAP, AIM GT which are 3D
        gt = np.random.choice(
            np.array([0, 1], dtype="i1"),
            size=(n_cnv_alleles, n_samples),
            replace=True,
            p=[1 - p_allele, p_allele],
        )

        # Create the GT dataset under calldata.
        calldata.create_dataset(name="GT", data=gt)

        # Create the samples dataset (again) for this contig.
        contig_grp.create_dataset(name="samples", data=samples, dtype=str)

        # Create the variants group for this contig.
        variants_grp = contig_grp.require_group("variants")

        # Simulate the CIEND and CIPOS arrays under variants.
        ciend = np.random.randint(low=0, high=13200, size=n_cnv_alleles)
        cipos = np.random.randint(low=0, high=37200, size=n_cnv_alleles)
        variants_grp.create_dataset(name="CIEND", data=ciend)
        variants_grp.create_dataset(name="CIPOS", data=cipos)

        # Simulate the unique ID strings under variants.
        # Note: this is quicker than generating unique random strings.
        len_str_n_sites = len(str(n_cnv_alleles))
        variant_IDs = [
            f"CNV_{contig}{str(i).zfill(len_str_n_sites)}"
            for i in range(1, n_cnv_alleles + 1)
        ]
        variants_grp.create_dataset(name="ID", data=variant_IDs)

        # Simulate the filters under variants.
        filter_pass = np.random.choice(
            [False, True], size=n_cnv_alleles, p=[1 - p_filter_pass, p_filter_pass]
        )
        filter_qMerge = np.random.choice(
            [False, True], size=n_cnv_alleles, p=[1 - p_filter_qMerge, p_filter_qMerge]
        )
        variants_grp.create_dataset(name="FILTER_PASS", data=filter_pass)
        variants_grp.create_dataset(name="FILTER_qMerge", data=filter_qMerge)

        # Simulate POS under variants.
        variants_grp.create_dataset(name="POS", data=allele_start_pos)

        # Simulate END under variants.
        variants_grp.create_dataset(name="END", data=allele_end_pos)

    zarr.consolidate_metadata(zarr_path)


def simulate_cnv_discordant_read_calls(zarr_path, metadata_path, contigs, contig_sizes):
    # zarr_path is the output path to the zarr store
    # metadata_path is the input path for the sample metadata
    # contigs is the list of contigs, e.g. Ag has ('2R', '3R', 'X')
    # contig_sizes is a dictionary of the sizes of the contigs in base pairs

    # {release}/cnv/{sample_set}/discordant_read_calls/zarr
    # - {contig}
    #   - calldata
    #     - GT [2D array] [int] [0 or 1 for n_variants for n_samples]
    #   - samples [1D array] [str for n_samples]
    #   - variants
    #      - END [1D array] [int for n_variants]
    #      - EndBreakpointMethod [1D array] [-1 to 2 for n_variants]
    #      - ID [1D array] [unique str for n_variants]
    #      - POS [1D array] [int for n_variants]
    #      - Region [1D array] [unique str for n_variants]
    #      - StartBreakpointMethod [1D array] [int] [-1 to 1 for n_variants]
    # - sample_coverage_variance [1D array] [float] [0 to 0.5 for n_samples]
    # - sample_is_high_variance [1D array] [bool] [True or False for n_samples]
    # - samples [1D array] [str for n_samples]

    # Get a random probability for a sample being high variance, between 0 and 1.
    p_variance = np.random.random()

    # Get a random probability for choosing allele 1, between 0 and 1.
    p_allele = np.random.random()

    # Open a zarr at the specified path.
    root = zarr.open(zarr_path, mode="w")

    # Create samples array.
    df_samples = pd.read_csv(metadata_path, engine="python")
    samples = df_samples["sample_id"].values
    root.create_dataset(name="samples", data=samples, dtype=str)

    # Get the number of samples.
    n_samples = len(df_samples)

    # Simulate sample_coverage_variance array.
    sample_coverage_variance = np.random.uniform(low=0, high=0.5, size=n_samples)
    root.create_dataset(name="sample_coverage_variance", data=sample_coverage_variance)

    # Simulate sample_is_high_variance array.
    sample_is_high_variance = np.random.choice(
        [False, True], size=n_samples, p=[1 - p_variance, p_variance]
    )
    root.create_dataset(name="sample_is_high_variance", data=sample_is_high_variance)

    # Note: The cnv_discordant_read_calls() method of AnophelesCnvData concatenates each xarray.Dataset
    # returned by _cnv_discordant_read_calls_dataset() for each sample set for each contig, so it expects
    # the same number of variants for all sample sets with respect to each contig. So we need to maintain a
    # consistent number of variants for each contig for all sample set, otherwise the shapes will not align,
    # which will raise an error and cause test failures.
    fixed_seed = 42

    for i, contig in enumerate(contigs):
        # Use the same random seed per contig, otherwise n_cnv_variants (and shapes) will not align.
        unique_seed = fixed_seed + i
        np.random.seed(unique_seed)

        # Create the contig group.
        contig_grp = root.require_group(contig)

        # Create the calldata group for this contig.
        calldata_grp = contig_grp.require_group("calldata")

        # Get the length of this contig
        contig_length_bp = contig_sizes[contig]

        # Get a random number of CNV variants to simulate.
        n_cnv_variants = np.random.randint(1, 100)

        # Produce a set of random start positions for each variant as a sorted list.
        variant_start_pos = sorted(
            np.random.randint(1, contig_length_bp, size=n_cnv_variants)
        )

        # Produce a set of random lengths for each variant, according to a range.
        variant_length_bp_min = 100
        variant_length_bp_max = 100_000
        variant_lengths_bp = np.random.randint(
            variant_length_bp_min, variant_length_bp_max, size=n_cnv_variants
        )

        # Produce the set of end postions for each variant, according to start position and length.
        variant_end_pos = [
            start_pos + length
            for start_pos, length in zip(variant_start_pos, variant_lengths_bp)
        ]

        # Simulate the genotype calls.
        # Note: this is only 2D, unlike SNP, HAP, AIM GT which are 3D
        gt = np.random.choice(
            np.array([0, 1], dtype="i1"),
            size=(n_cnv_variants, n_samples),
            replace=True,
            p=[1 - p_allele, p_allele],
        )

        # Create the GT dataset under calldata.
        calldata_grp.create_dataset(name="GT", data=gt)

        # Create the samples dataset (again) for this contig.
        contig_grp.create_dataset(name="samples", data=samples, dtype=str)

        # Create the variants group for this contig.
        variants_grp = contig_grp.require_group("variants")

        # Simulate the StartBreakpointMethod and EndBreakpointMethod arrays.
        startBreakpointMethod = np.random.randint(low=-1, high=1, size=n_cnv_variants)
        endBreakpointMethod = np.random.randint(low=-1, high=2, size=n_cnv_variants)
        variants_grp.create_dataset(
            name="StartBreakpointMethod", data=startBreakpointMethod
        )
        variants_grp.create_dataset(
            name="EndBreakpointMethod", data=endBreakpointMethod
        )

        # Get the number of digits in n_cnv_variants.
        len_str_n_cnv_variants = len(str(n_cnv_variants))

        # Simulate the Region under variants.
        # Note: this is quicker than generating unique random strings.
        regions = [
            f"Region_{str(i).zfill(len_str_n_cnv_variants)}"
            for i in range(1, n_cnv_variants + 1)
        ]
        variants_grp.create_dataset(name="Region", data=regions)

        # Simulate the unique ID strings under variants.
        # Note: this is quicker than generating unique random strings.
        variant_IDs = [
            f"ID_{str(i).zfill(len_str_n_cnv_variants)}"
            for i in range(1, n_cnv_variants + 1)
        ]
        variants_grp.create_dataset(name="ID", data=variant_IDs)

        # Simulate POS under variants.
        variants_grp.create_dataset(name="POS", data=variant_start_pos)

        # Simulate END under variants.
        variants_grp.create_dataset(name="END", data=variant_end_pos)

    zarr.consolidate_metadata(zarr_path)


class AnophelesSimulator:
    def __init__(
        self,
        fixture_dir: Path,
        bucket: str,
        releases: Tuple[str, ...],
        has_aims: bool,
        has_cohorts_by_quarter: bool,
        has_sequence_qc: bool,
    ):
        self.fixture_dir = fixture_dir
        self.bucket = bucket
        self.bucket_path = (self.fixture_dir / "simulated" / self.bucket).resolve()
        self.results_cache_path = (
            self.fixture_dir / "simulated" / "results_cache"
        ).resolve()
        self.url = self.bucket_path.as_uri()
        self.releases = releases
        self.has_aims = has_aims
        self.has_cohorts_by_quarter = has_cohorts_by_quarter
        self.has_sequence_qc = has_sequence_qc

        # Clear out the fixture directories.
        shutil.rmtree(self.bucket_path, ignore_errors=True)
        shutil.rmtree(self.results_cache_path, ignore_errors=True)

        # Ensure the fixture directory exists.
        self.bucket_path.mkdir(parents=True, exist_ok=True)

        # These members to be overridden/populated in subclasses.
        self.config: Dict[str, Any] = dict()
        self.contig_sizes: Dict[str, int] = dict()
        self.release_manifests: Dict[str, pd.DataFrame] = dict()
        self.genome = None
        self.genome_features = None

        # Create fixture data.
        self.init_config()
        self.init_public_release_manifest()
        self.init_pre_release_manifest()
        self.init_genome_sequence()
        self.init_genome_features()
        self.init_metadata()
        self.init_snp_sites()
        self.init_site_filters()
        self.init_snp_genotypes()
        self.init_site_annotations()
        self.init_hap_sites()
        self.init_haplotypes()
        self.init_aim_variants()
        self.init_aim_calls()
        self.init_cnv_hmm()
        self.init_cnv_coverage_calls()
        self.init_cnv_discordant_read_calls()

    @property
    def contigs(self) -> Tuple[str, ...]:
        return tuple(self.config["CONTIGS"])

    def random_contig(self):
        return choice(self.contigs)

    def random_transcript_id(self):
        df_transcripts = self.genome_features.query("type == 'mRNA'")
        transcript_ids = [
            _gff3_parse_attributes(t)["ID"] for t in df_transcripts.loc[:, "attributes"]
        ]
        transcript_id = choice(transcript_ids)
        return transcript_id

    def random_region_str(self, region_size=None):
        contig = self.random_contig()
        contig_size = self.contig_sizes[contig]
        region_start = randint(1, contig_size)
        if region_size:
            # Ensure we the region span doesn't exceed the contig size.
            if contig_size - region_start < region_size:
                region_start = contig_size - region_size

            region_end = region_start + region_size
        else:
            region_end = randint(region_start, contig_size)
        region = f"{contig}:{region_start:,}-{region_end:,}"
        return region

    # The following methods are overridden by subclasses.

    def init_config(self):
        pass

    def init_public_release_manifest(self):
        pass

    def init_pre_release_manifest(self):
        pass

    def init_genome_sequence(self):
        pass

    def init_genome_features(self):
        pass

    def init_metadata(self):
        pass

    def init_snp_sites(self):
        pass

    def init_site_filters(self):
        pass

    def init_snp_genotypes(self):
        pass

    def init_site_annotations(self):
        pass

    def init_hap_sites(self):
        pass

    def init_haplotypes(self):
        pass

    def init_aim_variants(self):
        pass

    def init_aim_calls(self):
        pass

    def init_cnv_hmm(self):
        pass

    def init_cnv_coverage_calls(self):
        pass

    def init_cnv_discordant_read_calls(self):
        pass


class Ag3Simulator(AnophelesSimulator):
    def __init__(self, fixture_dir):
        super().__init__(
            fixture_dir=fixture_dir,
            bucket="vo_agam_release_master_us_central1",
            releases=("3.0", "3.1"),
            has_aims=True,
            has_cohorts_by_quarter=True,
            has_sequence_qc=True,
        )

    def init_config(self):
        self.config = {
            "PUBLIC_RELEASES": ["3.0"],
            "GENESET_GFF3_PATH": "reference/genome/agamp4/Anopheles-gambiae-PEST_BASEFEATURES_AgamP4.12.gff3.gz",
            "GENOME_FASTA_PATH": "reference/genome/agamp4/Anopheles-gambiae-PEST_CHROMOSOMES_AgamP4.fa",
            "GENOME_FAI_PATH": "reference/genome/agamp4/Anopheles-gambiae-PEST_CHROMOSOMES_AgamP4.fa.fai",
            "GENOME_ZARR_PATH": "reference/genome/agamp4/Anopheles-gambiae-PEST_CHROMOSOMES_AgamP4.zarr",
            "GENOME_REF_ID": "AgamP4",
            "GENOME_REF_NAME": "Anopheles gambiae (PEST)",
            "CONTIGS": ["2R", "2L", "3R", "3L", "X"],
            "SITE_ANNOTATIONS_ZARR_PATH": "reference/genome/agamp4/Anopheles-gambiae-PEST_SEQANNOTATION_AgamP4.12.zarr",
            "DEFAULT_AIM_ANALYSIS": "20220528",
            "DEFAULT_SITE_FILTERS_ANALYSIS": "dt_20200416",
            "DEFAULT_COHORTS_ANALYSIS": "20230516",
            "SITE_MASK_IDS": ["gamb_colu_arab", "gamb_colu", "arab"],
            "PHASING_ANALYSIS_IDS": ["gamb_colu_arab", "gamb_colu", "arab"],
            "COVERAGE_CALLS_ANALYSIS_IDS": ["gamb_colu", "arab"],
            "DEFAULT_DISCORDANT_READ_CALLS_ANALYSIS": "20230911",
        }
        config_path = self.bucket_path / "v3-config.json"
        with config_path.open(mode="w") as f:
            json.dump(self.config, f, indent=4)

    def init_public_release_manifest(self):
        # Here we create a release manifest for an Ag3-style
        # public release. Note this is not the exact same data
        # as the real release.
        release_path = self.bucket_path / "v3"
        release_path.mkdir(parents=True, exist_ok=True)
        manifest_path = release_path / "manifest.tsv"
        manifest = pd.DataFrame(
            {
                "sample_set": ["AG1000G-AO", "AG1000G-BF-A"],
                "sample_count": [randint(10, 50), randint(10, 40)],
                "study_id": ["AG1000G-AO", "AG1000G-BF-1"],
                "study_url": [
                    "https://www.malariagen.net/network/where-we-work/AG1000G-AO",
                    "https://www.malariagen.net/network/where-we-work/AG1000G-BF-1",
                ],
                "terms_of_use_expiry_date": [
                    "2025-01-01",
                    "2025-01-01",
                ],
                "terms_of_use_url": [
                    "https://www.malariagen.net/data/our-approach-sharing-data/ag1000g-terms-of-use/",
                    "https://www.malariagen.net/data/our-approach-sharing-data/ag1000g-terms-of-use/",
                ],
            }
        )
        manifest.to_csv(manifest_path, index=False, sep="\t")
        self.release_manifests["3.0"] = manifest

    def init_pre_release_manifest(self):
        # Here we create a release manifest for an Ag3-style
        # pre-release. Note this is not the exact same data
        # as the real release.
        release_path = self.bucket_path / "v3.1"
        release_path.mkdir(parents=True, exist_ok=True)
        manifest_path = release_path / "manifest.tsv"
        manifest = pd.DataFrame(
            {
                "sample_set": [
                    "1177-VO-ML-LEHMANN-VMF00004",
                ],
                # Make sure we have some gambiae, coluzzii and arabiensis.
                "sample_count": [randint(20, 60)],
                "study_id": ["1177-VO-ML-LEHMANN"],
                "study_url": [
                    "https://www.malariagen.net/network/where-we-work/1177-VO-ML-LEHMANN"
                ],
                "terms_of_use_expiry_date": [
                    "2025-11-17",
                ],
                "terms_of_use_url": [
                    "https://malariagen.github.io/vector-data/ag3/ag3.1.html#terms-of-use",
                ],
            }
        )
        manifest.to_csv(manifest_path, index=False, sep="\t")
        self.release_manifests["3.1"] = manifest

    def init_genome_sequence(self):
        # Here we simulate a reference genome in a simple way
        # but with much smaller contigs. The data are stored
        # using zarr as with the real data releases.

        # Use real AgamP4 base composition.
        base_composition = {
            b"a": 0.042154199245128525,
            b"c": 0.027760739796444212,
            b"g": 0.027853725511269512,
            b"t": 0.041827104954587246,
            b"n": 0.028714045930701336,
            b"A": 0.23177421009505061,
            b"C": 0.1843981552034527,
            b"G": 0.1840007377851694,
            b"T": 0.23151655721224917,
            b"N": 5.242659472466922e-07,
        }
        path = self.bucket_path / self.config["GENOME_ZARR_PATH"]
        self.genome = simulate_genome(
            path=path,
            contigs=self.contigs,
            low=50_000,
            high=100_000,
            base_composition=base_composition,
        )
        self.contig_sizes = {
            contig: self.genome[contig].shape[0] for contig in self.contigs
        }

    def init_genome_features(self):
        path = self.bucket_path / self.config["GENESET_GFF3_PATH"]
        path.parent.mkdir(parents=True, exist_ok=True)
        simulator = Gff3Simulator(contig_sizes=self.contig_sizes)
        self.genome_features = simulator.simulate_gff(path=path)

    def write_metadata(
        self,
        release,
        release_path,
        sample_set,
        aim=True,
        cohorts=True,
        sequence_qc=True,
    ):
        # Here we take the approach of using some of the real metadata,
        # but truncating it to the number of samples included in the
        # simulated data resource.

        # Look up the number of samples in this sample set within the
        # simulated data resource.
        n_samples_sim = (
            self.release_manifests[release]
            .set_index("sample_set")
            .loc[sample_set]["sample_count"]
        )

        # Create general metadata by sampling from some real metadata files.
        src_path = (
            self.fixture_dir
            / "vo_agam_release_master_us_central1"
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "samples.meta.csv"
        )
        df_general = pd.read_csv(src_path, engine="python")
        # Randomly downsample.
        df_general_ds = df_general.sample(n_samples_sim, replace=False)
        samples_ds = df_general_ds["sample_id"].tolist()
        dst_path = (
            self.bucket_path
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "samples.meta.csv"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df_general_ds.to_csv(dst_path, index=False)

        # Create surveillance flags by sample from real metadata files.
        surv_flags_src_path = (
            self.fixture_dir
            / "vo_agam_release_master_us_central1"
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "surveillance.flags.csv"
        )
        df_surveillance_flags = pd.read_csv(surv_flags_src_path, engine="python")
        df_surveillance_flags_ds = (
            df_surveillance_flags.set_index("sample_id").loc[samples_ds].reset_index()
        )
        surv_flags_dst_path = (
            self.bucket_path
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "surveillance.flags.csv"
        )
        surv_flags_dst_path.parent.mkdir(parents=True, exist_ok=True)
        df_surveillance_flags_ds.to_csv(surv_flags_dst_path, index=False)

        if sequence_qc:
            # Create sequence QC metadata by sample from real metadata files.
            src_path = (
                self.fixture_dir
                / "vo_agam_release_master_us_central1"
                / release_path
                / "metadata"
                / "curation"
                / sample_set
                / "sequence_qc_stats.csv"
            )
            df_sequence_qc_stats = pd.read_csv(src_path, engine="python")
            df_sequence_qc_stats_ds = (
                df_sequence_qc_stats.set_index("sample_id")
                .loc[samples_ds]
                .reset_index()
            )
            dst_path = (
                self.bucket_path
                / release_path
                / "metadata"
                / "curation"
                / sample_set
                / "sequence_qc_stats.csv"
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            df_sequence_qc_stats_ds.to_csv(dst_path, index=False)

        if aim:
            # Create AIM metadata by sampling from some real metadata files.
            src_path = (
                self.fixture_dir
                / "vo_agam_release_master_us_central1"
                / release_path
                / "metadata"
                / "species_calls_aim_20220528"
                / sample_set
                / "samples.species_aim.csv"
            )
            df_aim = pd.read_csv(src_path, engine="python")
            df_aim_ds = df_aim.set_index("sample_id").loc[samples_ds].reset_index()
            dst_path = (
                self.bucket_path
                / release_path
                / "metadata"
                / "species_calls_aim_20220528"
                / sample_set
                / "samples.species_aim.csv"
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            df_aim_ds.to_csv(dst_path, index=False)

        if cohorts:
            # Create cohorts metadata by sampling from some real metadata files.
            src_path = (
                self.fixture_dir
                / "vo_agam_release_master_us_central1"
                / release_path
                / "metadata"
                / "cohorts_20230516"
                / sample_set
                / "samples.cohorts.csv"
            )
            df_coh = pd.read_csv(src_path, engine="python")
            df_coh_ds = df_coh.set_index("sample_id").loc[samples_ds].reset_index()
            dst_path = (
                self.bucket_path
                / release_path
                / "metadata"
                / "cohorts_20230516"
                / sample_set
                / "samples.cohorts.csv"
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            df_coh_ds.to_csv(dst_path, index=False)

            # Create cohorts data by sampling from some real files.
            src_path = (
                self.fixture_dir
                / "vo_agam_release_master_us_central1"
                / "v3_cohorts"
                / "cohorts_20230516"
                / "cohorts_admin1_month.csv"
            )
            dst_path = (
                self.bucket_path
                / "v3_cohorts"
                / "cohorts_20230516"
                / "cohorts_admin1_month.csv"
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with open(src_path, mode="r") as src, open(dst_path, mode="w") as dst:
                for line in src.readlines()[:5]:
                    print(line, file=dst)

        # Create data catalog by sampling from some real metadata files.
        src_path = (
            self.fixture_dir
            / "vo_agam_release_master_us_central1"
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "wgs_snp_data.csv"
        )
        df_cat = pd.read_csv(src_path, engine="python")
        df_cat_ds = df_cat.set_index("sample_id").loc[samples_ds].reset_index()
        dst_path = (
            self.bucket_path
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "wgs_snp_data.csv"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df_cat_ds.to_csv(dst_path, index=False)

        # Create accessions catalog by sampling from some real metadata files.
        src_path = (
            self.fixture_dir
            / "vo_agam_release_master_us_central1"
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "wgs_accession_data.csv"
        )
        df_cat = pd.read_csv(src_path, engine="python")
        df_cat_ds = df_cat.set_index("sample_id").loc[samples_ds].reset_index()
        dst_path = (
            self.bucket_path
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "wgs_accession_data.csv"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df_cat_ds.to_csv(dst_path, index=False)

    def init_metadata(self):
        self.write_metadata(release="3.0", release_path="v3", sample_set="AG1000G-AO")
        self.write_metadata(release="3.0", release_path="v3", sample_set="AG1000G-BF-A")
        self.write_metadata(
            release="3.1",
            release_path="v3.1",
            sample_set="1177-VO-ML-LEHMANN-VMF00004",
        )

    def init_snp_sites(self):
        path = self.bucket_path / "v3/snp_genotypes/all/sites/"
        self.snp_sites, self.n_snp_sites = simulate_snp_sites(
            path=path, contigs=self.contigs, genome=self.genome
        )

    def init_site_filters(self):
        analysis = self.config["DEFAULT_SITE_FILTERS_ANALYSIS"]

        # Simulate the gamb_colu mask.
        mask = "gamb_colu"
        p_pass = 0.71
        path = self.bucket_path / "v3/site_filters" / analysis / mask
        simulate_site_filters(
            path=path, contigs=self.contigs, p_pass=p_pass, n_sites=self.n_snp_sites
        )

        # Simulate the arab mask.
        mask = "arab"
        p_pass = 0.70
        path = self.bucket_path / "v3/site_filters" / analysis / mask
        simulate_site_filters(
            path=path, contigs=self.contigs, p_pass=p_pass, n_sites=self.n_snp_sites
        )

        # Simulate the gamb_colu_arab mask.
        mask = "gamb_colu_arab"
        p_pass = 0.62
        path = self.bucket_path / "v3/site_filters" / analysis / mask
        simulate_site_filters(
            path=path, contigs=self.contigs, p_pass=p_pass, n_sites=self.n_snp_sites
        )

    def init_snp_genotypes(self):
        # Iterate over releases.
        for release, manifest in self.release_manifests.items():
            # Determine release path.
            if release == "3.0":
                release_path = "v3"
            else:
                release_path = f"v{release}"

            # Iterate over sample sets in the release.
            for rec in manifest.itertuples():
                sample_set = rec.sample_set
                metadata_path = (
                    self.bucket_path
                    / release_path
                    / "metadata"
                    / "general"
                    / sample_set
                    / "samples.meta.csv"
                )

                # Create zarr hierarchy.
                zarr_path = (
                    self.bucket_path
                    / release_path
                    / "snp_genotypes"
                    / "all"
                    / sample_set
                )

                # Simulate SNP genotype data.
                p_allele = np.array([0.979, 0.007, 0.008, 0.006])
                p_missing = np.array([0.96, 0.04])
                simulate_snp_genotypes(
                    zarr_path=zarr_path,
                    metadata_path=metadata_path,
                    contigs=self.contigs,
                    n_sites=self.n_snp_sites,
                    p_allele=p_allele,
                    p_missing=p_missing,
                )

    def init_site_annotations(self):
        path = self.bucket_path / self.config["SITE_ANNOTATIONS_ZARR_PATH"]
        simulate_site_annotations(path=path, genome=self.genome)

    def init_hap_sites(self):
        self.hap_sites = dict()
        self.n_hap_sites = dict()
        analysis = "arab"
        path = self.bucket_path / "v3/snp_haplotypes/sites/" / analysis / "zarr"
        self.hap_sites[analysis], self.n_hap_sites[analysis] = simulate_hap_sites(
            path=path,
            contigs=self.contigs,
            snp_sites=self.snp_sites,
            p_site=0.09,
        )

        analysis = "gamb_colu"
        path = self.bucket_path / "v3/snp_haplotypes/sites/" / analysis / "zarr"
        self.hap_sites[analysis], self.n_hap_sites[analysis] = simulate_hap_sites(
            path=path,
            contigs=self.contigs,
            snp_sites=self.snp_sites,
            p_site=0.28,
        )

        analysis = "gamb_colu_arab"
        path = self.bucket_path / "v3/snp_haplotypes/sites/" / analysis / "zarr"
        self.hap_sites[analysis], self.n_hap_sites[analysis] = simulate_hap_sites(
            path=path,
            contigs=self.contigs,
            snp_sites=self.snp_sites,
            p_site=0.25,
        )

    def init_haplotypes(self):
        self.phasing_samples = dict()
        for release, manifest in self.release_manifests.items():
            # Determine release path.
            if release == "3.0":
                release_path = "v3"
            else:
                release_path = f"v{release}"

            # Iterate over sample sets in the release.
            for rec in manifest.itertuples():
                sample_set = rec.sample_set

                # Set up access to AIM metadata, to figure out which samples are in
                # which analysis.
                metadata_path = (
                    self.bucket_path
                    / release_path
                    / "metadata"
                    / "species_calls_aim_20220528"
                    / sample_set
                    / "samples.species_aim.csv"
                )
                df_aim = pd.read_csv(metadata_path, engine="python")

                # Simulate haplotypes for the gamb_colu_arab analysis.
                analysis = "gamb_colu_arab"
                p_1 = 0.008
                samples = df_aim["sample_id"].values
                self.phasing_samples[sample_set, analysis] = samples
                n_samples = len(samples)
                zarr_path = (
                    self.bucket_path
                    / release_path
                    / "snp_haplotypes"
                    / sample_set
                    / analysis
                    / "zarr"
                )
                root = zarr.open(zarr_path, mode="w")
                root.create_dataset(name="samples", data=samples, dtype=str)
                for contig in self.contigs:
                    n_sites = self.n_hap_sites[analysis][contig]
                    gt = np.random.choice(
                        np.array([0, 1], dtype="i1"),
                        size=(n_sites, n_samples, 2),
                        replace=True,
                        p=[1 - p_1, p_1],
                    )
                    calldata = root.require_group(contig).require_group("calldata")
                    calldata.create_dataset(name="GT", data=gt)
                zarr.consolidate_metadata(zarr_path)

                # Simulate haplotypes for the arab analysis.
                analysis = "arab"
                p_1 = 0.06
                samples = df_aim.query("aim_species == 'arabiensis'")[
                    "sample_id"
                ].values
                self.phasing_samples[sample_set, analysis] = samples
                n_samples = len(samples)
                if n_samples > 0:
                    zarr_path = (
                        self.bucket_path
                        / release_path
                        / "snp_haplotypes"
                        / sample_set
                        / analysis
                        / "zarr"
                    )
                    root = zarr.open(zarr_path, mode="w")
                    root.create_dataset(name="samples", data=samples, dtype=str)
                    for contig in self.contigs:
                        n_sites = self.n_hap_sites[analysis][contig]
                        gt = np.random.choice(
                            np.array([0, 1], dtype="i1"),
                            size=(n_sites, n_samples, 2),
                            replace=True,
                            p=[1 - p_1, p_1],
                        )
                        calldata = root.require_group(contig).require_group("calldata")
                        calldata.create_dataset(name="GT", data=gt)
                    zarr.consolidate_metadata(zarr_path)

                # Simulate haplotypes for the gamb_colu analysis.
                analysis = "gamb_colu"
                p_1 = 0.01
                samples = df_aim.query(
                    "aim_species not in ['arabiensis', 'intermediate_gambcolu_arabiensis']"
                )["sample_id"].values
                self.phasing_samples[sample_set, analysis] = samples
                n_samples = len(samples)
                if n_samples > 0:
                    zarr_path = (
                        self.bucket_path
                        / release_path
                        / "snp_haplotypes"
                        / sample_set
                        / analysis
                        / "zarr"
                    )
                    root = zarr.open(zarr_path, mode="w")
                    root.create_dataset(name="samples", data=samples, dtype=str)
                    for contig in self.contigs:
                        n_sites = self.n_hap_sites[analysis][contig]
                        gt = np.random.choice(
                            np.array([0, 1], dtype="i1"),
                            size=(n_sites, n_samples, 2),
                            replace=True,
                            p=[1 - p_1, p_1],
                        )
                        calldata = root.require_group(contig).require_group("calldata")
                        calldata.create_dataset(name="GT", data=gt)
                    zarr.consolidate_metadata(zarr_path)

    def init_aim_variants(self):
        self.aim_variants = dict()

        aims = "gambcolu_vs_arab"
        path = self.bucket_path / "reference/aim_defs_20220528" / f"{aims}.zarr"
        self.aim_variants[aims] = simulate_aim_variants(
            path=path,
            contigs=self.contigs,
            snp_sites=self.snp_sites,
            n_sites_low=1000,
            n_sites_high=4000,
        )

        aims = "gamb_vs_colu"
        path = self.bucket_path / "reference/aim_defs_20220528" / f"{aims}.zarr"
        self.aim_variants[aims] = simulate_aim_variants(
            path=path,
            contigs=self.contigs,
            snp_sites=self.snp_sites,
            n_sites_low=40,
            n_sites_high=400,
        )

    def init_aim_calls(self):
        # Iterate over releases.
        for release, manifest in self.release_manifests.items():
            # Determine release path.
            if release == "3.0":
                release_path = "v3"
            else:
                release_path = f"v{release}"

            # Iterate over sample sets in the release.
            for rec in manifest.itertuples():
                sample_set = rec.sample_set

                # Iterate over AIM sets.
                for aims in "gambcolu_vs_arab", "gamb_vs_colu":
                    ds_av = self.aim_variants[aims]

                    # Create AIM call dataset.
                    ds = ds_av.copy()

                    # Add sample_id variable.
                    metadata_path = (
                        self.bucket_path
                        / release_path
                        / "metadata"
                        / "general"
                        / sample_set
                        / "samples.meta.csv"
                    )
                    df_samples = pd.read_csv(metadata_path, engine="python")
                    ds["sample_id"] = ("samples",), df_samples["sample_id"]

                    # Add call_genotype variable.
                    gt = np.random.choice(
                        np.arange(2, dtype="i1"),
                        size=(ds.sizes["variants"], ds.sizes["samples"], 2),
                        replace=True,
                    )
                    ds["call_genotype"] = ("variants", "samples", "ploidy"), gt

                    # Write out zarr.
                    path = (
                        self.bucket_path
                        / f"{release_path}/aim_calls_20220528/{sample_set}/{aims}.zarr"
                    )
                    ds.to_zarr(path, mode="w", consolidated=True)

    def init_cnv_hmm(self):
        # Iterate over releases.
        for release, manifest in self.release_manifests.items():
            # Determine release path.
            if release == "3.0":
                release_path = "v3"
            else:
                release_path = f"v{release}"

            # Iterate over sample sets in the release.
            for rec in manifest.itertuples():
                sample_set = rec.sample_set
                metadata_path = (
                    self.bucket_path
                    / release_path
                    / "metadata"
                    / "general"
                    / sample_set
                    / "samples.meta.csv"
                )

                # Create zarr hierarchy.
                zarr_path = (
                    self.bucket_path
                    / release_path
                    / "cnv"
                    / sample_set
                    / "hmm"
                    / "zarr"
                )

                # Simulate CNV HMM data.
                simulate_cnv_hmm(
                    zarr_path=zarr_path,
                    metadata_path=metadata_path,
                    contigs=self.contigs,
                    contig_sizes=self.contig_sizes,
                )

    def init_cnv_coverage_calls(self):
        # Iterate over releases.
        for release, manifest in self.release_manifests.items():
            # Determine release path.
            if release == "3.0":
                release_path = "v3"
            else:
                release_path = f"v{release}"

            # Iterate over sample sets in the release.
            for rec in manifest.itertuples():
                sample_set = rec.sample_set
                metadata_path = (
                    self.bucket_path
                    / release_path
                    / "metadata"
                    / "general"
                    / sample_set
                    / "samples.meta.csv"
                )

                # Simulate CNV coverage calls data for the gamb_colu analysis.
                analysis = "gamb_colu"
                zarr_path = (
                    self.bucket_path
                    / release_path
                    / "cnv"
                    / sample_set
                    / "coverage_calls"
                    / analysis
                    / "zarr"
                )
                simulate_cnv_coverage_calls(
                    zarr_path=zarr_path,
                    metadata_path=metadata_path,
                    contigs=self.contigs,
                    contig_sizes=self.contig_sizes,
                )

                # Simulate CNV coverage calls data for the arab analysis.
                analysis = "arab"
                zarr_path = (
                    self.bucket_path
                    / release_path
                    / "cnv"
                    / sample_set
                    / "coverage_calls"
                    / analysis
                    / "zarr"
                )
                simulate_cnv_coverage_calls(
                    zarr_path=zarr_path,
                    metadata_path=metadata_path,
                    contigs=self.contigs,
                    contig_sizes=self.contig_sizes,
                )

    def init_cnv_discordant_read_calls(self):
        analysis = self.config["DEFAULT_DISCORDANT_READ_CALLS_ANALYSIS"]
        # Iterate over releases.
        for release, manifest in self.release_manifests.items():
            # Determine release path.
            if release == "3.0":
                release_path = "v3"
            else:
                release_path = f"v{release}"

            # Iterate over sample sets in the release.
            for rec in manifest.itertuples():
                sample_set = rec.sample_set
                metadata_path = (
                    self.bucket_path
                    / release_path
                    / "metadata"
                    / "general"
                    / sample_set
                    / "samples.meta.csv"
                )
                if analysis:
                    # Create zarr hierarchy.
                    zarr_path = (
                        self.bucket_path
                        / release_path
                        / "cnv"
                        / sample_set
                        / f"discordant_read_calls_{analysis}"
                        / "zarr"
                    )
                else:
                    # Create zarr hierarchy.
                    zarr_path = (
                        self.bucket_path
                        / release_path
                        / "cnv"
                        / sample_set
                        / "discordant_read_calls"
                        / "zarr"
                    )

                # Simulate CNV discordant read calls.
                simulate_cnv_discordant_read_calls(
                    zarr_path=zarr_path,
                    metadata_path=metadata_path,
                    # Note: the real data does not include every contig.
                    contigs=self.contigs,
                    contig_sizes=self.contig_sizes,
                )


class Af1Simulator(AnophelesSimulator):
    def __init__(self, fixture_dir):
        super().__init__(
            fixture_dir=fixture_dir,
            bucket="vo_afun_release_master_us_central1",
            releases=("1.0",),
            has_aims=False,
            has_cohorts_by_quarter=False,
            has_sequence_qc=True,
        )

    def init_config(self):
        self.config = {
            "PUBLIC_RELEASES": ["1.0"],
            "GENESET_GFF3_PATH": "reference/genome/idAnoFuneDA-416_04/VectorBase-61_AfunestusidAnoFuneDA416_04_patched.gff3.gz",
            "GENOME_FASTA_PATH": "reference/genome/idAnoFuneDA-416_04/idAnoFuneDA-416_04_1.curated_primary.fa",
            "GENOME_FAI_PATH": "reference/genome/idAnoFuneDA-416_04/idAnoFuneDA-416_04_1.curated_primary.fa.fai",
            "GENOME_ZARR_PATH": "reference/genome/idAnoFuneDA-416_04/idAnoFuneDA-416_04_1.curated_primary.zarr",
            "GENOME_REF_ID": "idAnoFuneDA-416_04",
            "GENOME_REF_NAME": "Anopheles funestus",
            "CONTIGS": ["2RL", "3RL", "X"],
            "SITE_ANNOTATIONS_ZARR_PATH": "reference/genome/idAnoFuneDA-416_04/Anopheles-funestus-DA-416_04_1_SEQANNOTATION.zarr",
            "DEFAULT_SITE_FILTERS_ANALYSIS": "dt_20200416",
            "DEFAULT_COHORTS_ANALYSIS": "20221129",
            "DEFAULT_DISCORDANT_READ_CALLS_ANALYSIS": "",
            "SITE_MASK_IDS": ["funestus"],
            "PHASING_ANALYSIS_IDS": ["funestus"],
            "COVERAGE_CALLS_ANALYSIS_IDS": ["funestus"],
        }
        config_path = self.bucket_path / "v1.0-config.json"
        with config_path.open(mode="w") as f:
            json.dump(self.config, f, indent=4)

    def init_public_release_manifest(self):
        # Here we create a release manifest for an Af1-style
        # public release. Note this is not the exact same data
        # as the real release.
        release_path = self.bucket_path / "v1.0"
        release_path.mkdir(parents=True, exist_ok=True)
        manifest_path = release_path / "manifest.tsv"
        manifest = pd.DataFrame(
            {
                "sample_set": [
                    "1229-VO-GH-DADZIE-VMF00095",
                    "1230-VO-GA-CF-AYALA-VMF00045",
                    "1231-VO-MULTI-WONDJI-VMF00043",
                    "1232-VO-KE-OCHOMO-VMF00044",
                    "1235-VO-MZ-PAAIJMANS-VMF00094",
                ],
                "sample_count": [26, 40, 32, 20, 20],
                "study_id": [
                    "1229-VO-GH-DADZIE",
                    "1230-VO-MULTI-AYALA",
                    "1231-VO-MULTI-WONDJI",
                    "1232-VO-KE-OCHOMO",
                    "1235-VO-MZ-PAAIJMANS",
                ],
                "study_url": [
                    "https://www.malariagen.net/network/where-we-work/1229-VO-GH-DADZIE",
                    "https://www.malariagen.net/network/where-we-work/1230-VO-MULTI-AYALA",
                    "https://www.malariagen.net/network/where-we-work/1231-VO-MULTI-WONDJI",
                    "https://www.malariagen.net/network/where-we-work/1232-VO-KE-OCHOMO",
                    "https://www.malariagen.net/network/where-we-work/1235-VO-MZ-PAAIJMANS",
                ],
                "terms_of_use_expiry_date": [
                    "2025-06-01",
                    "2025-06-01",
                    "2025-06-01",
                    "2024-01-01",  # Set to the past in order to test unrestricted_use_only.
                    "2024-01-01",  # Set to the past in order to test unrestricted_use_only. (We need at least 2 sets.)
                ],
                "terms_of_use_url": [
                    "https://malariagen.github.io/vector-data/af1/af1.0.html#terms-of-use",
                    "https://malariagen.github.io/vector-data/af1/af1.0.html#terms-of-use",
                    "https://malariagen.github.io/vector-data/af1/af1.0.html#terms-of-use",
                    "https://malariagen.github.io/vector-data/af1/af1.0.html#terms-of-use",
                    "https://malariagen.github.io/vector-data/af1/af1.0.html#terms-of-use",
                ],
            }
        )
        manifest.to_csv(manifest_path, index=False, sep="\t")
        self.release_manifests["1.0"] = manifest

    def init_genome_sequence(self):
        # Here we simulate a reference genome in a simple way
        # but with much smaller contigs. The data are stored
        # using zarr as with the real data releases.

        # Use real base composition.
        base_composition = {
            b"a": 0.0,
            b"c": 0.0,
            b"g": 0.0,
            b"t": 0.0,
            b"n": 0.0,
            b"A": 0.29432128333333335,
            b"C": 0.20542065,
            b"G": 0.20575796666666665,
            b"T": 0.2944834333333333,
            b"N": 1.6666666666666667e-05,
        }
        path = self.bucket_path / self.config["GENOME_ZARR_PATH"]
        self.genome = simulate_genome(
            path=path,
            contigs=self.contigs,
            low=80_000,
            high=120_000,
            base_composition=base_composition,
        )
        self.contig_sizes = {
            contig: self.genome[contig].shape[0] for contig in self.contigs
        }

    def init_genome_features(self):
        path = self.bucket_path / self.config["GENESET_GFF3_PATH"]
        path.parent.mkdir(parents=True, exist_ok=True)
        simulator = Gff3Simulator(
            contig_sizes=self.contig_sizes,
            # Af1 has a different gene type
            gene_type="protein_coding_gene",
            # Af1 has different attributes
            attrs=("Note", "description"),
        )
        self.genome_features = simulator.simulate_gff(path=path)

    def write_metadata(self, release, release_path, sample_set, sequence_qc=True):
        # Here we take the approach of using some of the real metadata,
        # but truncating it to the number of samples included in the
        # simulated data resource.

        # Look up the number of samples in this sample set within the
        # simulated data resource.
        n_samples_sim = (
            self.release_manifests[release]
            .set_index("sample_set")
            .loc[sample_set]["sample_count"]
        )

        # Create general metadata by sampling from some real metadata files.
        src_path = (
            self.fixture_dir
            / "vo_afun_release_master_us_central1"
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "samples.meta.csv"
        )
        df_general = pd.read_csv(src_path, engine="python")
        df_general_ds = df_general.sample(n_samples_sim, replace=False)
        samples_ds = df_general_ds["sample_id"].tolist()
        dst_path = (
            self.bucket_path
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "samples.meta.csv"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df_general_ds.to_csv(dst_path, index=False)

        # Create surveillance flags by sample from real metadata files.
        surv_flags_src_path = (
            self.fixture_dir
            / "vo_afun_release_master_us_central1"
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "surveillance.flags.csv"
        )
        df_surveillance_flags = pd.read_csv(surv_flags_src_path, engine="python")
        df_surveillance_flags_ds = (
            df_surveillance_flags.set_index("sample_id").loc[samples_ds].reset_index()
        )
        surv_flags_dst_path = (
            self.bucket_path
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "surveillance.flags.csv"
        )
        surv_flags_dst_path.parent.mkdir(parents=True, exist_ok=True)
        df_surveillance_flags_ds.to_csv(surv_flags_dst_path, index=False)

        if sequence_qc:
            # Create sequence QC metadata by sample from real metadata files.
            src_path = (
                self.fixture_dir
                / "vo_afun_release_master_us_central1"
                / release_path
                / "metadata"
                / "curation"
                / sample_set
                / "sequence_qc_stats.csv"
            )
            df_sequence_qc_stats = pd.read_csv(src_path, engine="python")
            df_sequence_qc_stats_ds = (
                df_sequence_qc_stats.set_index("sample_id")
                .loc[samples_ds]
                .reset_index()
            )
            dst_path = (
                self.bucket_path
                / release_path
                / "metadata"
                / "curation"
                / sample_set
                / "sequence_qc_stats.csv"
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            df_sequence_qc_stats_ds.to_csv(dst_path, index=False)

        # Create cohorts metadata by sampling from some real metadata files.
        src_path = (
            self.fixture_dir
            / "vo_afun_release_master_us_central1"
            / release_path
            / "metadata"
            / "cohorts_20221129"
            / sample_set
            / "samples.cohorts.csv"
        )
        df_coh = pd.read_csv(src_path, engine="python")
        df_coh_ds = df_coh.set_index("sample_id").loc[samples_ds].reset_index()
        dst_path = (
            self.bucket_path
            / release_path
            / "metadata"
            / "cohorts_20221129"
            / sample_set
            / "samples.cohorts.csv"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df_coh_ds.to_csv(dst_path, index=False)

        # Create data catalog by sampling from some real metadata files.
        src_path = (
            self.fixture_dir
            / "vo_afun_release_master_us_central1"
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "wgs_snp_data.csv"
        )
        df_cat = pd.read_csv(src_path, engine="python")
        df_cat_ds = df_cat.set_index("sample_id").loc[samples_ds].reset_index()
        dst_path = (
            self.bucket_path
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "wgs_snp_data.csv"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df_cat_ds.to_csv(dst_path, index=False)

        # Create accessions catalog by sampling from some real metadata files.
        src_path = (
            self.fixture_dir
            / "vo_afun_release_master_us_central1"
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "wgs_accession_data.csv"
        )
        df_cat = pd.read_csv(src_path, engine="python")
        df_cat_ds = df_cat.set_index("sample_id").loc[samples_ds].reset_index()
        dst_path = (
            self.bucket_path
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "wgs_accession_data.csv"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df_cat_ds.to_csv(dst_path, index=False)

    def init_metadata(self):
        self.write_metadata(
            release="1.0", release_path="v1.0", sample_set="1229-VO-GH-DADZIE-VMF00095"
        )
        self.write_metadata(
            release="1.0",
            release_path="v1.0",
            sample_set="1230-VO-GA-CF-AYALA-VMF00045",
        )
        self.write_metadata(
            release="1.0",
            release_path="v1.0",
            sample_set="1231-VO-MULTI-WONDJI-VMF00043",
        )
        self.write_metadata(
            release="1.0",
            release_path="v1.0",
            sample_set="1232-VO-KE-OCHOMO-VMF00044",
        )
        self.write_metadata(
            release="1.0",
            release_path="v1.0",
            sample_set="1235-VO-MZ-PAAIJMANS-VMF00094",
        )

    def init_snp_sites(self):
        path = self.bucket_path / "v1.0/snp_genotypes/all/sites/"
        self.snp_sites, self.n_snp_sites = simulate_snp_sites(
            path=path, contigs=self.contigs, genome=self.genome
        )

    def init_site_filters(self):
        analysis = self.config["DEFAULT_SITE_FILTERS_ANALYSIS"]

        # Simulate the funestus mask.
        mask = "funestus"
        p_pass = 0.59
        path = self.bucket_path / "v1.0/site_filters" / analysis / mask
        simulate_site_filters(
            path=path, contigs=self.contigs, p_pass=p_pass, n_sites=self.n_snp_sites
        )

    def init_snp_genotypes(self):
        # Iterate over releases.
        for release, manifest in self.release_manifests.items():
            # Determine release path.
            release_path = f"v{release}"

            # Iterate over sample sets in the release.
            for rec in manifest.itertuples():
                sample_set = rec.sample_set
                metadata_path = (
                    self.bucket_path
                    / release_path
                    / "metadata"
                    / "general"
                    / sample_set
                    / "samples.meta.csv"
                )

                # Create zarr hierarchy.
                zarr_path = (
                    self.bucket_path
                    / release_path
                    / "snp_genotypes"
                    / "all"
                    / sample_set
                )

                # Simulate SNP genotype data.
                p_allele = np.array([0.981, 0.006, 0.008, 0.005])
                p_missing = np.array([0.95, 0.05])
                simulate_snp_genotypes(
                    zarr_path=zarr_path,
                    metadata_path=metadata_path,
                    contigs=self.contigs,
                    n_sites=self.n_snp_sites,
                    p_allele=p_allele,
                    p_missing=p_missing,
                )

    def init_site_annotations(self):
        path = self.bucket_path / self.config["SITE_ANNOTATIONS_ZARR_PATH"]
        simulate_site_annotations(path=path, genome=self.genome)

    def init_hap_sites(self):
        self.hap_sites = dict()
        self.n_hap_sites = dict()
        analysis = "funestus"
        path = self.bucket_path / "v1.0/snp_haplotypes/sites/" / analysis / "zarr"
        self.hap_sites[analysis], self.n_hap_sites[analysis] = simulate_hap_sites(
            path=path,
            contigs=self.contigs,
            snp_sites=self.snp_sites,
            p_site=np.random.random(),
        )

    def init_haplotypes(self):
        self.phasing_samples = dict()
        for release, manifest in self.release_manifests.items():
            # Determine release path.
            release_path = f"v{release}"

            # Iterate over sample sets in the release.
            for rec in manifest.itertuples():
                sample_set = rec.sample_set

                # Access sample metadata to find samples.
                metadata_path = (
                    self.bucket_path
                    / release_path
                    / "metadata"
                    / "general"
                    / sample_set
                    / "samples.meta.csv"
                )
                df_samples = pd.read_csv(metadata_path, engine="python")
                samples = df_samples["sample_id"].values

                # Simulate haplotypes.
                analysis = "funestus"
                p_1 = np.random.random()
                samples = df_samples["sample_id"].values
                self.phasing_samples[sample_set, analysis] = samples
                n_samples = len(samples)
                zarr_path = (
                    self.bucket_path
                    / release_path
                    / "snp_haplotypes"
                    / sample_set
                    / analysis
                    / "zarr"
                )
                root = zarr.open(zarr_path, mode="w")
                root.create_dataset(name="samples", data=samples, dtype=str)
                for contig in self.contigs:
                    n_sites = self.n_hap_sites[analysis][contig]
                    gt = np.random.choice(
                        np.array([0, 1], dtype="i1"),
                        size=(n_sites, n_samples, 2),
                        replace=True,
                        p=[1 - p_1, p_1],
                    )
                    calldata = root.require_group(contig).require_group("calldata")
                    calldata.create_dataset(name="GT", data=gt)
                zarr.consolidate_metadata(zarr_path)

    def init_cnv_hmm(self):
        # Iterate over releases.
        for release, manifest in self.release_manifests.items():
            # Determine release path.
            release_path = f"v{release}"

            # Iterate over sample sets in the release.
            for rec in manifest.itertuples():
                sample_set = rec.sample_set
                metadata_path = (
                    self.bucket_path
                    / release_path
                    / "metadata"
                    / "general"
                    / sample_set
                    / "samples.meta.csv"
                )

                # Create zarr hierarchy.
                zarr_path = (
                    self.bucket_path
                    / release_path
                    / "cnv"
                    / sample_set
                    / "hmm"
                    / "zarr"
                )

                # Simulate CNV HMM data.
                simulate_cnv_hmm(
                    zarr_path=zarr_path,
                    metadata_path=metadata_path,
                    contigs=self.contigs,
                    contig_sizes=self.contig_sizes,
                )

    def init_cnv_coverage_calls(self):
        # Iterate over releases.
        for release, manifest in self.release_manifests.items():
            # Determine release path.
            release_path = f"v{release}"

            # Iterate over sample sets in the release.
            for rec in manifest.itertuples():
                sample_set = rec.sample_set
                metadata_path = (
                    self.bucket_path
                    / release_path
                    / "metadata"
                    / "general"
                    / sample_set
                    / "samples.meta.csv"
                )

                # Simulate CNV coverage calls data for the funestus analysis.
                analysis = "funestus"
                zarr_path = (
                    self.bucket_path
                    / release_path
                    / "cnv"
                    / sample_set
                    / "coverage_calls"
                    / analysis
                    / "zarr"
                )
                simulate_cnv_coverage_calls(
                    zarr_path=zarr_path,
                    metadata_path=metadata_path,
                    contigs=self.contigs,
                    contig_sizes=self.contig_sizes,
                )

    def init_cnv_discordant_read_calls(self):
        analysis = self.config["DEFAULT_DISCORDANT_READ_CALLS_ANALYSIS"]
        # Iterate over releases.
        for release, manifest in self.release_manifests.items():
            # Determine release path.
            release_path = f"v{release}"

            # Iterate over sample sets in the release.
            for rec in manifest.itertuples():
                sample_set = rec.sample_set
                metadata_path = (
                    self.bucket_path
                    / release_path
                    / "metadata"
                    / "general"
                    / sample_set
                    / "samples.meta.csv"
                )
                if analysis:
                    # Create zarr hierarchy.
                    zarr_path = (
                        self.bucket_path
                        / release_path
                        / "cnv"
                        / sample_set
                        / "discordant_read_calls_{analysis}"
                        / "zarr"
                    )
                else:
                    # Create zarr hierarchy.
                    zarr_path = (
                        self.bucket_path
                        / release_path
                        / "cnv"
                        / sample_set
                        / "discordant_read_calls"
                        / "zarr"
                    )

                # Simulate CNV discordant read calls.
                simulate_cnv_discordant_read_calls(
                    zarr_path=zarr_path,
                    metadata_path=metadata_path,
                    # Note: the real data does not include every contig.
                    contigs=self.contigs,
                    contig_sizes=self.contig_sizes,
                )


class Adir1Simulator(AnophelesSimulator):
    def __init__(self, fixture_dir):
        super().__init__(
            fixture_dir=fixture_dir,
            bucket="vo_adir_release_master_us_central1",
            releases=("1.0",),
            has_aims=False,
            has_cohorts_by_quarter=True,
            has_sequence_qc=True,
        )

    def init_config(self):
        self.config = {
            "PUBLIC_RELEASES": ["1.0"],
            "GENESET_GFF3_PATH": "reference/genome/AdirusWRAIR2/VectorBase-68_AdirusWRAIR2.gff.gz",
            "GENOME_FASTA_PATH": "reference/genome/AdirusWRAIR2/VectorBase-56_AdirusWRAIR2_Genome.fasta",
            "GENOME_FAI_PATH": "reference/genome/AdirusWRAIR2/VectorBase-56_AdirusWRAIR2_Genome.fasta.fai",
            "GENOME_ZARR_PATH": "reference/genome/AdirusWRAIR2/VectorBase-56_AdirusWRAIR2_Genome.zarr",
            "GENOME_REF_ID": "AdirusWRAIR2",
            "GENOME_REF_NAME": "Anopheles dirus",
            "CONTIGS": [
                "KB672490",
                "KB672868",
                "KB672979",
            ],  # Just using the three largest.
            "SITE_ANNOTATIONS_ZARR_PATH": "reference/genome/AdirusWRAIR2/VectorBase-56_AdirusWRAIR2_Genome.SEQANNOTATION.zarr",
            "DEFAULT_SITE_FILTERS_ANALYSIS": "sc_20250610",
            "DEFAULT_COHORTS_ANALYSIS": "20250710",
            "DEFAULT_DISCORDANT_READ_CALLS_ANALYSIS": "",
            "SITE_MASK_IDS": ["dirus"],
            "PHASING_ANALYSIS_IDS": ["dirus_noneyet"],
        }
        config_path = self.bucket_path / "v1.0-config.json"
        with config_path.open(mode="w") as f:
            json.dump(self.config, f, indent=4)

    def init_public_release_manifest(self):
        # Here we create a release manifest for an Adir1-style
        # public release. Note this is not the exact same data
        # as the real release.
        release_path = self.bucket_path / "v1.0"
        release_path.mkdir(parents=True, exist_ok=True)
        manifest_path = release_path / "manifest.tsv"
        manifest = pd.DataFrame(
            {
                "sample_set": [
                    "1277-VO-KH-WITKOWSKI-VMF00151",
                    "1276-AD-BD-ALAM-VMF00156",
                ],
                "sample_count": [20, 10],
                "study_id": [
                    "1277-VO-KH-WITKOWSKI",
                    "1276-AD-BD-ALAM",
                ],
                "study_url": [
                    "https://www.malariagen.net/network/where-we-work/1277-VO-KH-WITKOWSKI",
                    "https://www.malariagen.net/network/where-we-work/1276-AD-BD-ALAM",
                ],
                "terms_of_use_expiry_date": [
                    "2027-06-01",
                    "2027-06-01",
                ],
                "terms_of_use_url": [
                    "https://malariagen.github.io/vector-data/adir1/adir1.0.html#terms-of-use",
                    "https://malariagen.github.io/vector-data/adir1/adir1.0.html#terms-of-use",
                ],
            }
        )
        manifest.to_csv(manifest_path, index=False, sep="\t")
        self.release_manifests["1.0"] = manifest

    def init_genome_sequence(self):
        # Here we simulate a reference genome in a simple way
        # but with much smaller contigs. The data are stored
        # using zarr as with the real data releases.

        # Use real base composition.
        base_composition = {
            b"a": 0.0,
            b"c": 0.0,
            b"g": 0.0,
            b"t": 0.0,
            b"n": 0.0,
            b"A": 0.29432128333333335,
            b"C": 0.20542065,
            b"G": 0.20575796666666665,
            b"T": 0.2944834333333333,
            b"N": 1.6666666666666667e-05,
        }
        path = self.bucket_path / self.config["GENOME_ZARR_PATH"]
        self.genome = simulate_genome(
            path=path,
            contigs=self.contigs,
            low=80_000,
            high=120_000,
            base_composition=base_composition,
        )
        self.contig_sizes = {
            contig: self.genome[contig].shape[0] for contig in self.contigs
        }

    def init_genome_features(self):
        path = self.bucket_path / self.config["GENESET_GFF3_PATH"]
        path.parent.mkdir(parents=True, exist_ok=True)
        simulator = Gff3Simulator(
            contig_sizes=self.contig_sizes,
            # Adir1 has a different gene type
            gene_type="protein_coding_gene",
            # Adir1 has different attributes
            attrs=("Note", "description"),
        )
        self.genome_features = simulator.simulate_gff(path=path)

    def write_metadata(self, release, release_path, sample_set, sequence_qc=True):
        # Here we take the approach of using some of the real metadata,
        # but truncating it to the number of samples included in the
        # simulated data resource.

        # Look up the number of samples in this sample set within the
        # simulated data resource.
        n_samples_sim = (
            self.release_manifests[release]
            .set_index("sample_set")
            .loc[sample_set]["sample_count"]
        )

        # Create general metadata by sampling from some real metadata files.
        src_path = (
            self.fixture_dir
            / "vo_adir_release_master_us_central1"
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "samples.meta.csv"
        )
        df_general = pd.read_csv(src_path, engine="python")
        df_general_ds = df_general.sample(n_samples_sim, replace=False)
        samples_ds = df_general_ds["sample_id"].tolist()
        dst_path = (
            self.bucket_path
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "samples.meta.csv"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df_general_ds.to_csv(dst_path, index=False)

        if sequence_qc:
            # Create sequence QC metadata by sample from real metadata files.
            src_path = (
                self.fixture_dir
                / "vo_adir_release_master_us_central1"
                / release_path
                / "metadata"
                / "curation"
                / sample_set
                / "sequence_qc_stats.csv"
            )
            df_sequence_qc_stats = pd.read_csv(src_path, engine="python")
            df_sequence_qc_stats_ds = (
                df_sequence_qc_stats.set_index("sample_id")
                .loc[samples_ds]
                .reset_index()
            )
            dst_path = (
                self.bucket_path
                / release_path
                / "metadata"
                / "curation"
                / sample_set
                / "sequence_qc_stats.csv"
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            df_sequence_qc_stats_ds.to_csv(dst_path, index=False)

        # Create cohorts metadata by sampling from some real metadata files.
        src_path = (
            self.fixture_dir
            / "vo_adir_release_master_us_central1"
            / release_path
            / "metadata"
            / "cohorts_20250710"
            / sample_set
            / "samples.cohorts.csv"
        )
        df_coh = pd.read_csv(src_path, engine="python")
        df_coh_ds = df_coh.set_index("sample_id").loc[samples_ds].reset_index()
        dst_path = (
            self.bucket_path
            / release_path
            / "metadata"
            / "cohorts_20250710"
            / sample_set
            / "samples.cohorts.csv"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df_coh_ds.to_csv(dst_path, index=False)

        # Create data catalog by sampling from some real metadata files.
        src_path = (
            self.fixture_dir
            / "vo_adir_release_master_us_central1"
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "wgs_snp_data.csv"
        )
        df_cat = pd.read_csv(src_path, engine="python")
        df_cat_ds = df_cat.set_index("sample_id").loc[samples_ds].reset_index()
        dst_path = (
            self.bucket_path
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "wgs_snp_data.csv"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df_cat_ds.to_csv(dst_path, index=False)

        # Create accessions catalog by sampling from some real metadata files.
        src_path = (
            self.fixture_dir
            / "vo_adir_release_master_us_central1"
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "wgs_accession_data.csv"
        )
        df_cat = pd.read_csv(src_path, engine="python")
        df_cat_ds = df_cat.set_index("sample_id").loc[samples_ds].reset_index()
        dst_path = (
            self.bucket_path
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "wgs_accession_data.csv"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df_cat_ds.to_csv(dst_path, index=False)

    def init_metadata(self):
        self.write_metadata(
            release="1.0",
            release_path="v1.0",
            sample_set="1277-VO-KH-WITKOWSKI-VMF00151",
        )
        self.write_metadata(
            release="1.0",
            release_path="v1.0",
            sample_set="1276-AD-BD-ALAM-VMF00156",
        )

    def init_snp_sites(self):
        path = self.bucket_path / "v1.0/snp_genotypes/all/sites/"
        self.snp_sites, self.n_snp_sites = simulate_snp_sites(
            path=path, contigs=self.contigs, genome=self.genome
        )

    def init_site_filters(self):
        analysis = self.config["DEFAULT_SITE_FILTERS_ANALYSIS"]

        # Simulate the funestus mask.
        mask = "dirus"
        p_pass = 0.59
        path = self.bucket_path / "v1.0/site_filters" / analysis / mask
        simulate_site_filters(
            path=path, contigs=self.contigs, p_pass=p_pass, n_sites=self.n_snp_sites
        )

    def init_snp_genotypes(self):
        # Iterate over releases.
        for release, manifest in self.release_manifests.items():
            # Determine release path.
            release_path = f"v{release}"

            # Iterate over sample sets in the release.
            for rec in manifest.itertuples():
                sample_set = rec.sample_set
                metadata_path = (
                    self.bucket_path
                    / release_path
                    / "metadata"
                    / "general"
                    / sample_set
                    / "samples.meta.csv"
                )

                # Create zarr hierarchy.
                zarr_path = (
                    self.bucket_path
                    / release_path
                    / "snp_genotypes"
                    / "all"
                    / sample_set
                )

                # Simulate SNP genotype data.
                p_allele = np.array([0.981, 0.006, 0.008, 0.005])
                p_missing = np.array([0.95, 0.05])
                simulate_snp_genotypes(
                    zarr_path=zarr_path,
                    metadata_path=metadata_path,
                    contigs=self.contigs,
                    n_sites=self.n_snp_sites,
                    p_allele=p_allele,
                    p_missing=p_missing,
                )

    def init_site_annotations(self):
        path = self.bucket_path / self.config["SITE_ANNOTATIONS_ZARR_PATH"]
        simulate_site_annotations(path=path, genome=self.genome)


class Amin1Simulator(AnophelesSimulator):
    def __init__(self, fixture_dir):
        super().__init__(
            fixture_dir=fixture_dir,
            bucket="vo_amin_release_master_us_central1",
            releases=("1.0",),
            has_aims=False,
            has_cohorts_by_quarter=True,
            has_sequence_qc=True,
        )

    def init_config(self):
        self.config = {
            "PUBLIC_RELEASES": ["1.0"],
            "GENESET_GFF3_PATH": "reference/genome/aminm1/VectorBase-68_AminimusMINIMUS1.gff",
            "GENOME_FASTA_PATH": "reference/genome/aminm1/VectorBase-68_AminimusMINIMUS1_Genome.fasta",
            "GENOME_FAI_PATH": "reference/genome/aminm1/VectorBase-68_AminimusMINIMUS1_Genome.fasta.fai",
            "GENOME_ZARR_PATH": "reference/genome/aminm1/VectorBase-48_AminimusMINIMUS1_Genome.zarr",
            "GENOME_REF_ID": "AminimusMINIMUS1",
            "GENOME_REF_NAME": "Anopheles minimus",
            "CONTIGS": [
                "KB663610",
                "KB663721",
                "KB663832",
                "KB663943",
                "KB664054",
            ],  # Just using the three largest.
            "SITE_ANNOTATIONS_ZARR_PATH": "reference/aminm1/aminm1/Anopheles-minimus-MINIMUS1_SEQANNOTATION_AminM1.8.zarr",
            "DEFAULT_SITE_FILTERS_ANALYSIS": "sc_20250610",
            "DEFAULT_COHORTS_ANALYSIS": "20251019",
            "DEFAULT_DISCORDANT_READ_CALLS_ANALYSIS": "",
            "SITE_MASK_IDS": ["minimus"],
            "PHASING_ANALYSIS_IDS": ["minimus_noneyet"],
        }
        config_path = self.bucket_path / "v1.0-config.json"
        with config_path.open(mode="w") as f:
            json.dump(self.config, f, indent=4)

    def init_public_release_manifest(self):
        # Here we create a release manifest for an Adir1-style
        # public release. Note this is not the exact same data
        # as the real release.
        release_path = self.bucket_path / "v1.0"
        release_path.mkdir(parents=True, exist_ok=True)
        manifest_path = release_path / "manifest.tsv"
        manifest = pd.DataFrame(
            {
                "sample_set": [
                    "1175-VO-KH-STLAURENT",
                ],
                "sample_count": [20],
                "study_id": [
                    "1175-VO-KH-STLAURENT",
                ],
                "study_url": [
                    "https://www.malariagen.net/network/where-we-work/1175-VO-KH-STLAURENT",
                ],
                "terms_of_use_expiry_date": [
                    "2023-11-16",
                ],
                "terms_of_use_url": [
                    "https://malariagen.github.io/vector-data/amin1/amin1.0.html#terms-of-use",
                ],
            }
        )
        manifest.to_csv(manifest_path, index=False, sep="\t")
        self.release_manifests["1.0"] = manifest

    def init_genome_sequence(self):
        # Here we simulate a reference genome in a simple way
        # but with much smaller contigs. The data are stored
        # using zarr as with the real data releases.

        # Use real base composition.
        base_composition = {
            b"a": 0.0,
            b"c": 0.0,
            b"g": 0.0,
            b"t": 0.0,
            b"n": 0.0,
            b"A": 0.29432128333333335,
            b"C": 0.20542065,
            b"G": 0.20575796666666665,
            b"T": 0.2944834333333333,
            b"N": 1.6666666666666667e-05,
        }
        path = self.bucket_path / self.config["GENOME_ZARR_PATH"]
        self.genome = simulate_genome(
            path=path,
            contigs=self.contigs,
            low=80_000,
            high=120_000,
            base_composition=base_composition,
        )
        self.contig_sizes = {
            contig: self.genome[contig].shape[0] for contig in self.contigs
        }

    def init_genome_features(self):
        path = self.bucket_path / self.config["GENESET_GFF3_PATH"]
        path.parent.mkdir(parents=True, exist_ok=True)
        simulator = Gff3Simulator(
            contig_sizes=self.contig_sizes,
            # Amin1 has a different gene type
            gene_type="protein_coding_gene",
            # Amin1 has different attributes
            attrs=("Note", "description"),
        )
        self.genome_features = simulator.simulate_gff(path=path)

    def write_metadata(self, release, release_path, sample_set, sequence_qc=True):
        # Here we take the approach of using some of the real metadata,
        # but truncating it to the number of samples included in the
        # simulated data resource.

        # Look up the number of samples in this sample set within the
        # simulated data resource.
        n_samples_sim = (
            self.release_manifests[release]
            .set_index("sample_set")
            .loc[sample_set]["sample_count"]
        )

        # Create general metadata by sampling from some real metadata files.
        src_path = (
            self.fixture_dir
            / "vo_amin_release_master_us_central1"
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "samples.meta.csv"
        )
        df_general = pd.read_csv(src_path, engine="python")
        df_general_ds = df_general.sample(n_samples_sim, replace=False)
        samples_ds = df_general_ds["sample_id"].tolist()
        dst_path = (
            self.bucket_path
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "samples.meta.csv"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df_general_ds.to_csv(dst_path, index=False)

        if sequence_qc:
            # Create sequence QC metadata by sample from real metadata files.
            src_path = (
                self.fixture_dir
                / "vo_amin_release_master_us_central1"
                / release_path
                / "metadata"
                / "curation"
                / sample_set
                / "sequence_qc_stats.csv"
            )
            df_sequence_qc_stats = pd.read_csv(src_path, engine="python")
            df_sequence_qc_stats_ds = (
                df_sequence_qc_stats.set_index("sample_id")
                .loc[samples_ds]
                .reset_index()
            )
            dst_path = (
                self.bucket_path
                / release_path
                / "metadata"
                / "curation"
                / sample_set
                / "sequence_qc_stats.csv"
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            df_sequence_qc_stats_ds.to_csv(dst_path, index=False)

        # Create cohorts metadata by sampling from some real metadata files.
        src_path = (
            self.fixture_dir
            / "vo_amin_release_master_us_central1"
            / release_path
            / "metadata"
            / "cohorts_20251019"
            / sample_set
            / "samples.cohorts.csv"
        )
        df_coh = pd.read_csv(src_path, engine="python")
        df_coh_ds = df_coh.set_index("sample_id").loc[samples_ds].reset_index()
        dst_path = (
            self.bucket_path
            / release_path
            / "metadata"
            / "cohorts_20251019"
            / sample_set
            / "samples.cohorts.csv"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df_coh_ds.to_csv(dst_path, index=False)

        # Create data catalog by sampling from some real metadata files.
        src_path = (
            self.fixture_dir
            / "vo_amin_release_master_us_central1"
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "wgs_snp_data.csv"
        )
        df_cat = pd.read_csv(src_path, engine="python")
        df_cat_ds = df_cat.set_index("sample_id").loc[samples_ds].reset_index()
        dst_path = (
            self.bucket_path
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "wgs_snp_data.csv"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df_cat_ds.to_csv(dst_path, index=False)

        # Create accessions catalog by sampling from some real metadata files.
        src_path = (
            self.fixture_dir
            / "vo_amin_release_master_us_central1"
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "wgs_accession_data.csv"
        )
        df_cat = pd.read_csv(src_path, engine="python")
        df_cat_ds = df_cat.set_index("sample_id").loc[samples_ds].reset_index()
        dst_path = (
            self.bucket_path
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "wgs_accession_data.csv"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df_cat_ds.to_csv(dst_path, index=False)

    def init_metadata(self):
        self.write_metadata(
            release="1.0",
            release_path="v1.0",
            sample_set="1175-VO-KH-STLAURENT",
        )

    def init_snp_sites(self):
        path = self.bucket_path / "v1.0/snp_genotypes/all/sites/"
        self.snp_sites, self.n_snp_sites = simulate_snp_sites(
            path=path, contigs=self.contigs, genome=self.genome
        )

    def init_site_filters(self):
        analysis = self.config["DEFAULT_SITE_FILTERS_ANALYSIS"]

        # Simulate the minimus mask.
        mask = "minimus"
        p_pass = 0.59
        path = self.bucket_path / "v1.0/site_filters" / analysis / mask
        simulate_site_filters(
            path=path, contigs=self.contigs, p_pass=p_pass, n_sites=self.n_snp_sites
        )

    def init_snp_genotypes(self):
        # Iterate over releases.
        for release, manifest in self.release_manifests.items():
            # Determine release path.
            release_path = f"v{release}"

            # Iterate over sample sets in the release.
            for rec in manifest.itertuples():
                sample_set = rec.sample_set
                metadata_path = (
                    self.bucket_path
                    / release_path
                    / "metadata"
                    / "general"
                    / sample_set
                    / "samples.meta.csv"
                )

                # Create zarr hierarchy.
                zarr_path = (
                    self.bucket_path
                    / release_path
                    / "snp_genotypes"
                    / "all"
                    / sample_set
                )

                # Simulate SNP genotype data.
                p_allele = np.array([0.981, 0.006, 0.008, 0.005])
                p_missing = np.array([0.95, 0.05])
                simulate_snp_genotypes(
                    zarr_path=zarr_path,
                    metadata_path=metadata_path,
                    contigs=self.contigs,
                    n_sites=self.n_snp_sites,
                    p_allele=p_allele,
                    p_missing=p_missing,
                )

    def init_site_annotations(self):
        path = self.bucket_path / self.config["SITE_ANNOTATIONS_ZARR_PATH"]
        simulate_site_annotations(path=path, genome=self.genome)


# For the following data fixtures we will use the "session" scope
# so that the fixture data will be created only once per test
# session (i.e., per invocation of pytest).
#
# Recreating these test data ensures that any change to the code
# here which creates the fixtures will immediately be reflected
# in a change to the fixture data.
#
# However, only recreating the data once per test session minimises
# the amount of work needed to create the data.


@pytest.fixture(scope="session")
def ag3_sim_fixture(fixture_dir):
    return Ag3Simulator(fixture_dir=fixture_dir)


@pytest.fixture(scope="session")
def af1_sim_fixture(fixture_dir):
    return Af1Simulator(fixture_dir=fixture_dir)


@pytest.fixture(scope="session")
def adir1_sim_fixture(fixture_dir):
    return Adir1Simulator(fixture_dir=fixture_dir)


@pytest.fixture(scope="session")
def amin1_sim_fixture(fixture_dir):
    return Amin1Simulator(fixture_dir=fixture_dir)
