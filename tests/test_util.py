"""Tests for Region.__repr__ and CacheMiss.__repr__ / default message."""

import pytest

from unittest.mock import MagicMock, patch

from malariagen_data.util import CacheMiss, Region, _get_file_stats

# ---------------------------------------------------------------------------
# Region
# ---------------------------------------------------------------------------


def test_region_repr_contig_only():
    r = Region("2L")
    assert repr(r) == "Region('2L', None, None)"
    assert str(r) == "2L"


def test_region_repr_with_coords():
    r = Region("2L", 100_000, 200_000)
    assert repr(r) == "Region('2L', 100000, 200000)"
    assert str(r) == "2L:100,000-200,000"


def test_region_repr_in_list():
    regions = [Region("2L", 10, 20), Region("3R", 30, 40)]
    assert repr(regions) == "[Region('2L', 10, 20), Region('3R', 30, 40)]"


def test_region_repr_start_only():
    r = Region("X", start=500, end=None)
    assert repr(r) == "Region('X', 500, None)"
    assert str(r) == "X:500-"


# ---------------------------------------------------------------------------
# CacheMiss
# ---------------------------------------------------------------------------


def test_cache_miss_no_key():
    cm = CacheMiss()
    assert repr(cm) == "CacheMiss()"
    assert "Cache miss" in str(cm)


def test_cache_miss_string_key():
    cm = CacheMiss("my_key")
    assert repr(cm) == "CacheMiss('my_key')"
    assert "my_key" in str(cm)


def test_cache_miss_tuple_key():
    cm = CacheMiss(("contig", 100))
    assert repr(cm) == "CacheMiss(('contig', 100))"
    assert "('contig', 100)" in str(cm)


def test_cache_miss_is_exception():
    with pytest.raises(CacheMiss) as exc_info:
        raise CacheMiss("lookup_key")
    assert "lookup_key" in str(exc_info.value)
    assert repr(exc_info.value) == "CacheMiss('lookup_key')"

def test_get_file_stats_local(tmp_path):
    # Setup a local test file
    content = b"test content"
    p = tmp_path / "test.txt"
    p.write_bytes(content)

    # Test retrieval
    stats = _get_file_stats(str(p))

    assert stats["size"] == len(content)
    assert isinstance(stats["mtime"], (float, int))
    assert stats["protocol"] in ["file", "local"]
    assert stats["path"] == str(p)

def test_get_file_stats_missing_size():
    # Mock filesystem to return None for size
    mock_fs = MagicMock()
    mock_fs.info.return_value = {"size": None}
    mock_fs.protocol = "gs"

    with patch("malariagen_data.util._init_filesystem", return_value=(mock_fs, "dummy/path")):
        with pytest.raises(ValueError, match="Could not determine size for file"):
            _get_file_stats("gs://bucket/file")

def test_get_file_stats_protocol_normalization():
    # Mock filesystem with a list of protocols (common in fsspec)
    mock_fs = MagicMock()
    mock_fs.info.return_value = {"size": 100, "mtime": 123.4}
    mock_fs.protocol = ("s3", "s3a")

    with patch("malariagen_data.util._init_filesystem", return_value=(mock_fs, "dummy/path")):
        stats = _get_file_stats("s3://bucket/file")
        assert stats["protocol"] == "s3"

def test_get_file_stats_file_not_found():
    # Verify standard FileNotFoundError propagation
    with pytest.raises(FileNotFoundError):
        _get_file_stats("non_existent_file_9999.txt")
