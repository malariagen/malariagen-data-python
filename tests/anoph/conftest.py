import json
import shutil
from pathlib import Path

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

cwd = Path(__file__).parent.resolve()


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


class Ag3Fixture:
    def __init__(self):
        self.bucket = "vo_agam_release"
        self.path = (cwd / "fixture" / self.bucket).resolve()
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
            "DEFAULT_SPECIES_ANALYSIS": "aim_20220528",
            "DEFAULT_SITE_FILTERS_ANALYSIS": "dt_20200416",
            "DEFAULT_COHORTS_ANALYSIS": "20230223",
            "SITE_MASK_IDS": ["gamb_colu_arab", "gamb_colu", "arab"],
            "PHASING_ANALYSIS_IDS": ["gamb_colu_arab", "gamb_colu", "arab"],
        }
        config_path = self.path / "v3-config.json"
        with config_path.open(mode="w") as f:
            json.dump(self.config, f)

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
                "sample_count": [31, 42],
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
                    "1177-VO-ML-LEHMANN-VMF00015",
                    "1237-VO-BJ-DJOGBENOU-VMF00050",
                ],
                "sample_count": [23, 27],
            }
        )
        manifest.to_csv(manifest_path, index=False, sep="\t")
        self.release_manifests["3.1"] = manifest

    def init_genome_sequence(self):
        # Here we simulate a reference genome in a simple way
        # but with much smaller contigs. The data are stored
        # using zarr as with the real data releases.
        path = self.path / self.config["GENOME_ZARR_PATH"]
        simulate_genome(path=path, contigs=self.contigs, low=100_000, high=200_000)


class Af1Fixture:
    def __init__(self):
        self.bucket = "vo_afun_release"
        self.path = (cwd / "fixture" / self.bucket).resolve()
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
            json.dump(self.config, f)

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
        simulate_genome(path=path, contigs=self.contigs, low=100_000, high=300_000)


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
def ag3_fixture():
    return Ag3Fixture()


@pytest.fixture(scope="session")
def af1_fixture():
    return Af1Fixture()
