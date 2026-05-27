"""Tests for Region.__repr__ and CacheMiss.__repr__ / default message."""

import pytest

from malariagen_data.util import CacheMiss, Region


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
