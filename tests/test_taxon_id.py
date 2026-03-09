import gzip
import pytest
from malariagen_data import identify_taxon


@pytest.fixture
def mock_fastq(tmp_path):
    """Creates a mock gzipped FASTQ file for testing."""
    fastq_data = [
        "@seq1",
        "ATGCGTACGTTAGCTAGCTAGCTA",
        "+",
        "########################",
        "@seq2",
        "ATGCGTACGTTAGCTAGCTAGCTT",
        "+",
        "########################",
    ]

    fastq_file = tmp_path / "test.fastq.gz"
    with gzip.open(fastq_file, "wt") as f:
        f.write("\n".join(fastq_data) + "\n")

    return str(fastq_file)


def test_identify_taxon(mock_fastq):
    """Tests if the identify_taxon function outputs the expected JSON-like dict."""
    result = identify_taxon(mock_fastq)

    assert isinstance(result, dict)
    assert "predicted_group" in result
    assert "recommended_resource" in result
    assert "confidence" in result
    assert "method" in result
    assert "top_candidates" in result

    assert result["method"] == "kmer_screening"
    assert result["predicted_group"] in [
        "gambiae_complex",
        "funestus_subgroup",
        "stephensi",
    ]

    # Based on the mock sequences in the fixture, it should strongly predict gambiae_complex
    assert result["predicted_group"] == "gambiae_complex"
    assert result["recommended_resource"] == "Ag3"

    candidates = result["top_candidates"]
    assert len(candidates) == 3
    assert candidates[0]["taxon"] == "gambiae_complex"


def test_identify_taxon_missing_file():
    """Tests error handling for missing files."""
    with pytest.raises(ValueError, match="Failed to read or parse fastq file"):
        identify_taxon("does_not_exist.fastq.gz")
