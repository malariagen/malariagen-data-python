import json
import shutil
import string
from pathlib import Path
from random import choice, choices, randint
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest
import zarr

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
        intron_size_high=1_000,
        exon_size_low=10,
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
        for exon_ix in range(randint(self.n_exons_low, self.n_exons_high)):
            exon_id = f"exon-{contig}-{gene_ix}-{transcript_ix}-{exon_ix}"
            intron_size = randint(
                self.intron_size_low, min(transcript_size, self.intron_size_high)
            )
            exon_start = exon_end + intron_size
            if exon_start >= transcript_end:
                # Stop making exons, no more space left in the transcript.
                break
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

        # Note that this is not perfect, because sometimes we end up
        # without any CDSs. Also in reality, an exon can contain
        # part of a UTR and part of a CDS, but that is harder to
        # simulate. So keep things simple for now.
        if strand == "-":
            # Take exons in reverse order.
            exons == exons[::-1]
        for exon_ix, exon in enumerate(exons):
            first_exon = exon_ix == 0
            last_exon = exon_ix == len(exons) - 1
            if first_exon:
                feature_type = self.utr5_type
                phase = "."
            elif last_exon:
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
    return n_sites


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
    df_samples = pd.read_csv(metadata_path)
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


class AnophelesSimulator:
    def __init__(
        self,
        fixture_dir: Path,
        bucket: str,
        releases: Tuple[str, ...],
        has_aims: bool,
        has_cohorts_by_quarter: bool,
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

    @property
    def contigs(self) -> Tuple[str, ...]:
        return tuple(self.config["CONTIGS"])

    def random_contig(self):
        return choice(self.contigs)

    def random_region_str(self):
        contig = self.random_contig()
        contig_size = self.contig_sizes[contig]
        region_start = randint(1, contig_size)
        region_end = randint(region_start, contig_size)
        region = f"{contig}:{region_start:,}-{region_end:,}"
        return region

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


class Ag3Simulator(AnophelesSimulator):
    def __init__(self, fixture_dir):
        super().__init__(
            fixture_dir=fixture_dir,
            bucket="vo_agam_release",
            releases=("3.0", "3.1"),
            has_aims=True,
            has_cohorts_by_quarter=True,
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
            "DEFAULT_COHORTS_ANALYSIS": "20230223",
            "SITE_MASK_IDS": ["gamb_colu_arab", "gamb_colu", "arab"],
            "PHASING_ANALYSIS_IDS": ["gamb_colu_arab", "gamb_colu", "arab"],
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
                "sample_count": [randint(10, 60), randint(10, 50)],
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
                "sample_count": [randint(10, 70)],
            }
        )
        manifest.to_csv(manifest_path, index=False, sep="\t")
        self.release_manifests["3.1"] = manifest

    def init_genome_sequence(self):
        # Here we simulate a reference genome in a simple way
        # but with much smaller contigs. The data are stored
        # using zarr as with the real data releases.

        # Use real base composition.
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
            low=100_000,
            high=150_000,
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

    def write_metadata(self, release, release_path, sample_set, aim=True, cohorts=True):
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
            / "vo_agam_release"
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "samples.meta.csv"
        )
        dst_path = (
            self.bucket_path
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "samples.meta.csv"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(src_path, mode="r") as src, open(dst_path, mode="w") as dst:
            for line in src.readlines()[: n_samples_sim + 1]:
                print(line, file=dst)

        if aim:
            # Create AIM metadata by sampling from some real metadata files.
            src_path = (
                self.fixture_dir
                / "vo_agam_release"
                / release_path
                / "metadata"
                / "species_calls_aim_20220528"
                / sample_set
                / "samples.species_aim.csv"
            )
            dst_path = (
                self.bucket_path
                / release_path
                / "metadata"
                / "species_calls_aim_20220528"
                / sample_set
                / "samples.species_aim.csv"
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with open(src_path, mode="r") as src, open(dst_path, mode="w") as dst:
                for line in src.readlines()[: n_samples_sim + 1]:
                    print(line, file=dst)

        if cohorts:
            # Create cohorts metadata by sampling from some real metadata files.
            src_path = (
                self.fixture_dir
                / "vo_agam_release"
                / release_path
                / "metadata"
                / "cohorts_20230223"
                / sample_set
                / "samples.cohorts.csv"
            )
            dst_path = (
                self.bucket_path
                / release_path
                / "metadata"
                / "cohorts_20230223"
                / sample_set
                / "samples.cohorts.csv"
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with open(src_path, mode="r") as src, open(dst_path, mode="w") as dst:
                for line in src.readlines()[: n_samples_sim + 1]:
                    print(line, file=dst)

        # Create data catalog by sampling from some real metadata files.
        src_path = (
            self.fixture_dir
            / "vo_agam_release"
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "wgs_snp_data.csv"
        )
        dst_path = (
            self.bucket_path
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "wgs_snp_data.csv"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(src_path, mode="r") as src, open(dst_path, mode="w") as dst:
            for line in src.readlines()[: n_samples_sim + 1]:
                print(line, file=dst)

    def init_metadata(self):
        self.write_metadata(release="3.0", release_path="v3", sample_set="AG1000G-AO")
        self.write_metadata(release="3.0", release_path="v3", sample_set="AG1000G-BF-A")
        # Simulate situation where AIM and cohorts metadata are missing,
        # do we correctly fill?
        self.write_metadata(
            release="3.1",
            release_path="v3.1",
            sample_set="1177-VO-ML-LEHMANN-VMF00004",
            aim=False,
            cohorts=False,
        )

    def init_snp_sites(self):
        path = self.bucket_path / "v3/snp_genotypes/all/sites/"
        self.n_sites = simulate_snp_sites(
            path=path, contigs=self.contigs, genome=self.genome
        )

    def init_site_filters(self):
        analysis = self.config["DEFAULT_SITE_FILTERS_ANALYSIS"]

        # Simulate the gamb_colu mask.
        mask = "gamb_colu"
        p_pass = 0.71
        path = self.bucket_path / "v3/site_filters" / analysis / mask
        simulate_site_filters(
            path=path, contigs=self.contigs, p_pass=p_pass, n_sites=self.n_sites
        )

        # Simulate the arab mask.
        mask = "arab"
        p_pass = 0.70
        path = self.bucket_path / "v3/site_filters" / analysis / mask
        simulate_site_filters(
            path=path, contigs=self.contigs, p_pass=p_pass, n_sites=self.n_sites
        )

        # Simulate the gamb_colu_arab mask.
        mask = "gamb_colu_arab"
        p_pass = 0.62
        path = self.bucket_path / "v3/site_filters" / analysis / mask
        simulate_site_filters(
            path=path, contigs=self.contigs, p_pass=p_pass, n_sites=self.n_sites
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
                    n_sites=self.n_sites,
                    p_allele=p_allele,
                    p_missing=p_missing,
                )

    def init_site_annotations(self):
        path = self.bucket_path / self.config["SITE_ANNOTATIONS_ZARR_PATH"]
        simulate_site_annotations(path=path, genome=self.genome)


class Af1Simulator(AnophelesSimulator):
    def __init__(self, fixture_dir):
        super().__init__(
            fixture_dir=fixture_dir,
            bucket="vo_afun_release",
            releases=("1.0",),
            has_aims=False,
            has_cohorts_by_quarter=False,
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
            "SITE_MASK_IDS": ["funestus"],
            "PHASING_ANALYSIS_IDS": ["funestus"],
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
                ],
                "sample_count": [36, 50, 32],
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
            low=100_000,
            high=200_000,
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

    def write_metadata(self, release, release_path, sample_set):
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
            / "vo_afun_release"
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "samples.meta.csv"
        )
        dst_path = (
            self.bucket_path
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "samples.meta.csv"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(src_path, mode="r") as src, open(dst_path, mode="w") as dst:
            for line in src.readlines()[: n_samples_sim + 1]:
                print(line, file=dst)

        # Create cohorts metadata by sampling from some real metadata files.
        src_path = (
            self.fixture_dir
            / "vo_afun_release"
            / release_path
            / "metadata"
            / "cohorts_20221129"
            / sample_set
            / "samples.cohorts.csv"
        )
        dst_path = (
            self.bucket_path
            / release_path
            / "metadata"
            / "cohorts_20221129"
            / sample_set
            / "samples.cohorts.csv"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(src_path, mode="r") as src, open(dst_path, mode="w") as dst:
            for line in src.readlines()[: n_samples_sim + 1]:
                print(line, file=dst)

        # Create data catalog by sampling from some real metadata files.
        src_path = (
            self.fixture_dir
            / "vo_afun_release"
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "wgs_snp_data.csv"
        )
        dst_path = (
            self.bucket_path
            / release_path
            / "metadata"
            / "general"
            / sample_set
            / "wgs_snp_data.csv"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(src_path, mode="r") as src, open(dst_path, mode="w") as dst:
            for line in src.readlines()[: n_samples_sim + 1]:
                print(line, file=dst)

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

    def init_snp_sites(self):
        path = self.bucket_path / "v1.0/snp_genotypes/all/sites/"
        self.n_sites = simulate_snp_sites(
            path=path, contigs=self.contigs, genome=self.genome
        )

    def init_site_filters(self):
        analysis = self.config["DEFAULT_SITE_FILTERS_ANALYSIS"]

        # Simulate the funestus mask.
        mask = "funestus"
        p_pass = 0.59
        path = self.bucket_path / "v1.0/site_filters" / analysis / mask
        simulate_site_filters(
            path=path, contigs=self.contigs, p_pass=p_pass, n_sites=self.n_sites
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
                    n_sites=self.n_sites,
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
