from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from .conftest import (
    Gff3Simulator,
    simulate_cnv_discordant_read_calls,
    simulate_cnv_hmm,
)


def _write_sample_metadata(path: Path, n_samples: int = 100) -> None:
    df_samples = pd.DataFrame({"sample_id": [f"S{i:04d}" for i in range(n_samples)]})
    df_samples.to_csv(path, index=False)


def test_simulate_cnv_hmm_limits_high_variance_fraction(tmp_path):
    zarr_path = tmp_path / "cnv_hmm.zarr"
    metadata_path = tmp_path / "samples.csv"
    _write_sample_metadata(metadata_path)

    simulate_cnv_hmm(
        zarr_path=zarr_path,
        metadata_path=metadata_path,
        contigs=("2L",),
        contig_sizes={"2L": 10_000},
        rng=np.random.default_rng(0),
    )

    root = zarr.open(zarr_path, mode="r")
    high_variance_fraction = np.mean(root["sample_is_high_variance"][:])
    assert high_variance_fraction < 0.3


def test_simulate_cnv_discordant_read_calls_limits_high_variance_fraction(tmp_path):
    zarr_path = tmp_path / "cnv_discordant.zarr"
    metadata_path = tmp_path / "samples.csv"
    _write_sample_metadata(metadata_path)

    simulate_cnv_discordant_read_calls(
        zarr_path=zarr_path,
        metadata_path=metadata_path,
        contigs=("2L",),
        contig_sizes={"2L": 10_000},
        rng=np.random.default_rng(0),
    )

    root = zarr.open(zarr_path, mode="r")
    high_variance_fraction = np.mean(root["sample_is_high_variance"][:])
    assert high_variance_fraction < 0.3


def test_simulate_exons_on_minus_strand_reverses_feature_order():
    sim = Gff3Simulator(
        contig_sizes={"2L": 10_000},
        rng=np.random.default_rng(0),
        n_exons_low=3,
        n_exons_high=3,
        intron_size_low=10,
        intron_size_high=10,
        exon_size_low=100,
        exon_size_high=100,
    )
    rows = list(
        sim.simulate_exons(
            contig="2L",
            strand="-",
            gene_ix=0,
            transcript_ix=0,
            transcript_id="transcript-2L-0-0",
            transcript_start=1,
            transcript_end=1_000,
        )
    )
    cds_and_utrs = [
        row for row in rows if row[2] in {sim.utr5_type, sim.utr3_type, sim.cds_type}
    ]
    starts = [row[3] for row in cds_and_utrs]
    assert starts == sorted(starts, reverse=True)
