import gzip
import shutil
import subprocess

import pytest

from malariagen_data.bgzf import _MAX_BLOCK_UNCOMPRESSED_SIZE, bgzf_open


def test_bgzf_round_trip(tmp_path):
    output_path = str(tmp_path / "test.txt.gz")
    lines = [f"line {i}\n" for i in range(100)]

    with bgzf_open(output_path, "wt") as f:
        for line in lines:
            f.write(line)

    # BGZF is valid gzip, so a plain gzip reader must be able to read
    # the exact content back.
    with gzip.open(output_path, "rt") as f:
        content = f.read()
    assert content == "".join(lines)


def test_bgzf_multi_block_round_trip(tmp_path):
    # Write enough data to force multiple BGZF blocks, and include
    # content that lands exactly on, just under, and just over a block
    # boundary, to exercise the block-splitting logic.
    output_path = str(tmp_path / "test_multiblock.txt.gz")
    line = "the quick brown fox jumps over the lazy dog 0123456789\n"
    n_lines = (_MAX_BLOCK_UNCOMPRESSED_SIZE // len(line)) * 3 + 7
    expected = "".join(f"{i}:{line}" for i in range(n_lines))

    with bgzf_open(output_path, "wt") as f:
        for i in range(n_lines):
            f.write(f"{i}:{line}")

    with gzip.open(output_path, "rt") as f:
        content = f.read()
    assert content == expected


def test_bgzf_empty_file(tmp_path):
    # Writing nothing should still produce a valid (if empty) BGZF
    # file, i.e. just the EOF marker.
    output_path = str(tmp_path / "test_empty.txt.gz")
    with bgzf_open(output_path, "wt"):
        pass

    with gzip.open(output_path, "rt") as f:
        content = f.read()
    assert content == ""


def test_bgzf_rejects_binary_mode(tmp_path):
    output_path = str(tmp_path / "test.txt.gz")
    with pytest.raises(ValueError):
        bgzf_open(output_path, "wb")


@pytest.mark.skipif(shutil.which("bgzip") is None, reason="bgzip not installed")
def test_bgzf_valid_per_bgzip(tmp_path):
    # The definitive check: the reference BGZF implementation itself
    # must accept this as valid BGZF, not just "some gzip stream that
    # happens to decompress correctly."
    output_path = str(tmp_path / "test.txt.gz")
    with bgzf_open(output_path, "wt") as f:
        for i in range(10_000):
            f.write(f"line {i}\n")

    result = subprocess.run(
        ["bgzip", "-t", output_path], capture_output=True, text=True
    )
    assert result.returncode == 0, f"not valid BGZF: {result.stderr}"


@pytest.mark.skipif(shutil.which("tabix") is None, reason="tabix not installed")
def test_bgzf_tabix_indexable(tmp_path):
    # tabix specifically requires genuine BGZF block structure to
    # build an index at all; plain gzip fails here even though it
    # decompresses fine. This is the property that made bedtools (and
    # other BGZF-aware tools) unable to read plain-gzip VCF output
    # correctly, even though bcftools tolerated it.
    output_path = str(tmp_path / "test.vcf.gz")
    with bgzf_open(output_path, "wt") as f:
        f.write("##fileformat=VCFv4.3\n")
        f.write("##contig=<ID=2R>\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for pos in (100, 200, 300):
            f.write(f"2R\t{pos}\t.\tA\tT\t.\t.\t.\n")

    result = subprocess.run(
        ["tabix", "-p", "vcf", output_path], capture_output=True, text=True
    )
    assert result.returncode == 0, f"tabix failed to index: {result.stderr}"
