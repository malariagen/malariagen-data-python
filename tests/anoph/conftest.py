import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

# We are going to create some data locally which follows
# the same layout and format of the real data in GCS,
# but which is much smaller and so can be used for faster
# test runs. This data is referred to here as the
# "data fixture".


cwd = Path(__file__).parent.resolve()


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
                "sample_count": [81, 181],
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
                "sample_count": [23, 90],
            }
        )
        manifest.to_csv(manifest_path, index=False, sep="\t")
        self.release_manifests["3.1"] = manifest


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
        self.init_config()
        self.init_public_release_manifest()

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
                "sample_count": [36, 50, 320],
            }
        )
        manifest.to_csv(manifest_path, index=False, sep="\t")


@pytest.fixture(scope="session")
def ag3_fixture():
    return Ag3Fixture()


@pytest.fixture(scope="session")
def af1_fixture():
    return Af1Fixture()
