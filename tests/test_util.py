import pytest
from malariagen_data.util import _parse_single_region, Region, CacheMiss
from unittest.mock import MagicMock

# _parse_single_region tests

@pytest.fixture
def mock_resource():
    resource = MagicMock()
    resource.contigs = ("2R", "2L", "3R", "3L", "X")
    del resource.virtual_contigs
    resource.genome_sequence.return_value.shape = [1_000_000]
    return resource

def test_parse_region_contig(mock_resource):
    r = _parse_single_region(mock_resource, "2L")
    assert r.contig == "2L"
    assert r.start is None
    assert r.end is None

    r = _parse_single_region(mock_resource, {"contig":"2L"})
    assert r.contig == "2L"
    assert r.start is None
    assert r.end is None

def test_parse_region_interval(mock_resource):
    r = _parse_single_region(mock_resource, "2L:100-200")
    assert r.contig == "2L"
    assert r.start == 100
    assert r.end == 200

def test_parse_region_invalid_string(mock_resource):
    invalid_regions = ["invalid_region", 
                       "3L:abc-100", 
                       "2L:100-25d", 
                       "", 
                       "2R:100-50", 
                       "3R:0-10", 
                       "2R-100-200", 
                       "2L:150"] 
    for region in invalid_regions:
        with pytest.raises(ValueError):
            _parse_single_region(mock_resource, region)

def test_parse_region_invalid_dictionary(mock_resource):
    invalid_regions = [{}, 
                       {"start": 100, "end":200}, 
                       {"contig":"3L", "start":-2, "end":10}, 
                       {"contig":"X", "start":10, "end":-100}, 
                       {"contig":"2L", "start":100, "end":10}, 
                       {"contig": "Invalid_contig", "start":10, "end":20}, 
                       {"contig":"2L", "start":"abc", "end":10}, 
                       {"contig":"2R", "start":100, "end":"bcd"}
                       ]
    
    for region in invalid_regions:
        with pytest.raises(ValueError):
            _parse_single_region(mock_resource, region)
    
def test_parse_region_dictionary_only_start(mock_resource):
    r = _parse_single_region(mock_resource, {"contig": "2L", "start":200})
    assert r.contig == "2L"
    assert r.start == 200
    assert r.end is None

def test_parse_region_dictionary_only_end(mock_resource):
    r = _parse_single_region(mock_resource, {"contig": "2L", "end":200})
    assert r.contig == "2L"
    assert r.start is None
    assert r.end == 200
            
def test_parse_region_invalid_type(mock_resource):
    invalid_types = [123456, ["2L:100-230"], ("2L:100-250",), True, False, 3.154]
    for types in invalid_types:
        with pytest.raises(TypeError):
            _parse_single_region(mock_resource, types)

#  Region tests

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

# CacheMiss tests

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
