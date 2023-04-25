import json
import shutil
import string
from pathlib import Path
from random import choice, choices, randint

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


def simulate_contig(*, low, high):
    size = np.random.randint(low=low, high=high)
    seq = np.random.choice(
        [b"A", b"C", b"G", b"T", b"N", b"a", b"c", b"g", b"t", b"n"],
        size=size,
    )
    return seq


def simulate_genome(*, path, contigs, low, high):
    path.mkdir(parents=True, exist_ok=True)
    root = zarr.open(path, mode="w")
    for contig in contigs:
        seq = simulate_contig(low=low, high=high)
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


class Ag3Simulator:
    def __init__(self, fixture_dir):
        self.fixture_dir = fixture_dir
        self.bucket = "vo_agam_release"
        self.path = (self.fixture_dir / "simulated" / self.bucket).resolve()
        self.url = self.path.as_uri()

        # Clear out the fixture directory.
        shutil.rmtree(self.path, ignore_errors=True)

        # Ensure the fixture directory exists.
        self.path.mkdir(parents=True, exist_ok=True)

        # Create fixture data.
        self.releases = ("3.0", "3.1")
        self.release_manifests = dict()
        self.init_config()
        self.init_public_release_manifest()
        self.init_pre_release_manifest()
        self.init_genome_sequence()
        self.init_genome_features()
        self.init_metadata()

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
        config_path = self.path / "v3-config.json"
        with config_path.open(mode="w") as f:
            json.dump(self.config, f, indent=4)

    @property
    def contigs(self):
        return tuple(self.config["CONTIGS"])

    def init_public_release_manifest(self):
        # Here we create a release manifest for an Ag3-style
        # public release. Note this is not the exact same data
        # as the real release.
        release_path = self.path / "v3"
        release_path.mkdir(parents=True, exist_ok=True)
        manifest_path = release_path / "manifest.tsv"
        manifest = pd.DataFrame(
            {
                "sample_set": ["AG1000G-AO", "AG1000G-BF-A"],
                "sample_count": [randint(10, 81), randint(10, 100)],
            }
        )
        manifest.to_csv(manifest_path, index=False, sep="\t")
        self.release_manifests["3.0"] = manifest

    def init_pre_release_manifest(self):
        # Here we create a release manifest for an Ag3-style
        # pre-release. Note this is not the exact same data
        # as the real release.
        release_path = self.path / "v3.1"
        release_path.mkdir(parents=True, exist_ok=True)
        manifest_path = release_path / "manifest.tsv"
        manifest = pd.DataFrame(
            {
                "sample_set": [
                    "1177-VO-ML-LEHMANN-VMF00004",
                ],
                "sample_count": [randint(10, 100)],
            }
        )
        manifest.to_csv(manifest_path, index=False, sep="\t")
        self.release_manifests["3.1"] = manifest

    def init_genome_sequence(self):
        # Here we simulate a reference genome in a simple way
        # but with much smaller contigs. The data are stored
        # using zarr as with the real data releases.
        # TODO Use accurate base composition.
        path = self.path / self.config["GENOME_ZARR_PATH"]
        self.genome = simulate_genome(
            path=path, contigs=self.contigs, low=100_000, high=200_000
        )
        self.contig_sizes = {
            contig: self.genome[contig].shape[0] for contig in self.contigs
        }

    def init_genome_features(self):
        path = self.path / self.config["GENESET_GFF3_PATH"]
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
            self.path
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
                self.path
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
                self.path
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
            self.path
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


class Af1Simulator:
    def __init__(self, fixture_dir):
        self.fixture_dir = fixture_dir
        self.bucket = "vo_afun_release"
        self.path = (self.fixture_dir / "simulated" / self.bucket).resolve()
        self.url = self.path.as_uri()

        # Clear out the fixture directory.
        shutil.rmtree(self.path, ignore_errors=True)

        # Ensure the fixture directory exists.
        self.path.mkdir(parents=True, exist_ok=True)

        # Create fixture data.
        self.releases = ("1.0",)
        self.release_manifests = dict()
        self.init_config()
        self.init_public_release_manifest()
        self.init_genome_sequence()
        self.init_genome_features()
        self.init_metadata()

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
        config_path = self.path / "v1.0-config.json"
        with config_path.open(mode="w") as f:
            json.dump(self.config, f, indent=4)

    @property
    def contigs(self):
        return tuple(self.config["CONTIGS"])

    def init_public_release_manifest(self):
        # Here we create a release manifest for an Af1-style
        # public release. Note this is not the exact same data
        # as the real release.
        release_path = self.path / "v1.0"
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
        path = self.path / self.config["GENOME_ZARR_PATH"]
        self.genome = simulate_genome(
            path=path, contigs=self.contigs, low=100_000, high=300_000
        )
        self.contig_sizes = {
            contig: self.genome[contig].shape[0] for contig in self.contigs
        }

    def init_genome_features(self):
        path = self.path / self.config["GENESET_GFF3_PATH"]
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
            self.path
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
            self.path
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
            self.path
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
