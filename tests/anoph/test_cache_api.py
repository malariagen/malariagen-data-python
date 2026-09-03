import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.base import AnophelesBase


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesBase(
        url=ag3_sim_fixture.url,
        public_url=ag3_sim_fixture.url,
        config_path=_ag3.CONFIG_PATH,
        major_version_number=_ag3.MAJOR_VERSION_NUMBER,
        major_version_path=_ag3.MAJOR_VERSION_PATH,
        pre=True,
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesBase(
        url=af1_sim_fixture.url,
        public_url=af1_sim_fixture.url,
        config_path=_af1.CONFIG_PATH,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
    )


def case_ag3_sim(ag3_sim_fixture, ag3_sim_api):
    return ag3_sim_fixture, ag3_sim_api


def case_af1_sim(af1_sim_fixture, af1_sim_api):
    return af1_sim_fixture, af1_sim_api


@parametrize_with_cases("fixture,api", cases=".")
def test_cache_info_returns_dict(fixture, api):
    """cache_info() should return a dict with the expected structure."""
    info = api.cache_info()
    assert isinstance(info, dict)

    for attr_name, entry in info.items():
        assert isinstance(attr_name, str)
        assert "entries" in entry
        assert "nbytes" in entry
        assert "kind" in entry
        assert "note" in entry
        assert isinstance(entry["entries"], int)
        assert isinstance(entry["nbytes"], (int, float))
        assert entry["kind"] in ("dict", "lru_cache", "other")
        assert isinstance(entry["note"], str)


@parametrize_with_cases("fixture,api", cases=".")
def test_cache_info_after_population(fixture, api):
    """cache_info() should reflect entries after a cache is populated."""
    # Populate the base sample_sets cache by calling sample_sets().
    api.sample_sets()

    info = api.cache_info()

    # After calling sample_sets(), _cache_sample_sets should have entries.
    if "_cache_sample_sets" in info:
        assert info["_cache_sample_sets"]["entries"] > 0
        assert info["_cache_sample_sets"]["kind"] == "dict"


@parametrize_with_cases("fixture,api", cases=".")
def test_clear_cache_all(fixture, api):
    """clear_cache('all') should empty all dict caches."""
    # Populate some caches.
    api.sample_sets()

    # Verify something is cached.
    info_before = api.cache_info()
    has_entries = any(
        v["entries"] > 0 for v in info_before.values() if v["kind"] == "dict"
    )
    assert has_entries, "Expected at least one populated dict cache"

    # Clear all.
    api.clear_cache()

    # All dict caches should now be empty.
    info_after = api.cache_info()
    for attr_name, entry in info_after.items():
        if entry["kind"] == "dict":
            assert (
                entry["entries"] == 0
            ), f"Cache {attr_name} still has {entry['entries']} entries after clear_cache()"


@parametrize_with_cases("fixture,api", cases=".")
def test_clear_cache_specific_category(fixture, api):
    """clear_cache with a specific category should only clear that category."""
    # Populate the base caches.
    api.sample_sets()

    # Clear only the "base" category.
    api.clear_cache("base")

    info = api.cache_info()

    # All "base" category caches should be empty.
    for attr_name in AnophelesBase._CACHE_CATEGORIES["base"]:
        if attr_name in info and info[attr_name]["kind"] == "dict":
            assert (
                info[attr_name]["entries"] == 0
            ), f"Cache {attr_name} should be empty after clear_cache('base')"


@parametrize_with_cases("fixture,api", cases=".")
def test_clear_cache_invalid_category(fixture, api):
    """clear_cache with an invalid category should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown cache category"):
        api.clear_cache("nonexistent")


@parametrize_with_cases("fixture,api", cases=".")
def test_clear_cache_repopulates_on_demand(fixture, api):
    """After clear_cache(), accessing data should repopulate the cache."""
    # Populate.
    api.sample_sets()

    # Clear.
    api.clear_cache()

    # Access again — should work and repopulate.
    df = api.sample_sets()
    assert isinstance(df, pd.DataFrame)

    info = api.cache_info()
    if "_cache_sample_sets" in info:
        assert info["_cache_sample_sets"]["entries"] > 0


def test_cache_info_size_estimation():
    """Test the static size estimation helper with known types."""
    # numpy array
    arr = np.zeros((100, 100), dtype=np.float64)
    nbytes, note = AnophelesBase._estimate_cache_entry_nbytes(arr)
    assert nbytes == arr.nbytes
    assert note == "numpy.nbytes"

    # xarray Dataset
    ds = xr.Dataset({"var": xr.DataArray(np.zeros((50, 50), dtype=np.float32))})
    nbytes, note = AnophelesBase._estimate_cache_entry_nbytes(ds)
    assert nbytes == ds.nbytes
    assert note == "xarray.nbytes"

    # bytes
    b = b"hello world"
    nbytes, note = AnophelesBase._estimate_cache_entry_nbytes(b)
    assert nbytes == len(b)
    assert note == "bytes length"

    # fallback
    obj = {"key": "value"}
    nbytes, note = AnophelesBase._estimate_cache_entry_nbytes(obj)
    assert nbytes > 0
    assert note == "sys.getsizeof shallow"


def test_clear_cache_direct_dict_manipulation():
    """Test cache clearing with directly manipulated dict caches."""
    # Create a minimal AnophelesBase-like object to test the mechanism
    # without needing full API setup.

    class FakeBase:
        _CACHE_CATEGORIES = AnophelesBase._CACHE_CATEGORIES

        def __init__(self):
            self._cache_haplotypes = {}
            self._cache_haplotype_sites = {}
            self._cache_cnv_hmm = {}
            self._cache_sample_metadata = {}

        # Bind the methods from AnophelesBase.
        _iter_cache_attrs = AnophelesBase._iter_cache_attrs
        _estimate_cache_entry_nbytes = staticmethod(
            AnophelesBase._estimate_cache_entry_nbytes
        )
        cache_info = AnophelesBase.cache_info
        clear_cache = AnophelesBase.clear_cache

    fake = FakeBase()

    # Populate caches with dummy data.
    fake._cache_haplotypes["key1"] = np.zeros((10, 10))
    fake._cache_haplotypes["key2"] = np.zeros((20, 20))
    fake._cache_haplotype_sites["key1"] = np.zeros((5, 5))
    fake._cache_cnv_hmm["key1"] = np.zeros((8, 8))
    fake._cache_sample_metadata["key1"] = pd.DataFrame({"a": [1, 2, 3]})

    # Check cache_info reports entries.
    info = fake.cache_info()
    assert info["_cache_haplotypes"]["entries"] == 2
    assert info["_cache_haplotype_sites"]["entries"] == 1
    assert info["_cache_cnv_hmm"]["entries"] == 1
    assert info["_cache_sample_metadata"]["entries"] == 1

    # Clear only haplotypes category.
    fake.clear_cache("haplotypes")
    info = fake.cache_info()
    assert info["_cache_haplotypes"]["entries"] == 0
    assert info["_cache_haplotype_sites"]["entries"] == 0
    # CNV and sample_metadata should be untouched.
    assert info["_cache_cnv_hmm"]["entries"] == 1
    assert info["_cache_sample_metadata"]["entries"] == 1

    # Clear all remaining.
    fake.clear_cache("all")
    info = fake.cache_info()
    for entry in info.values():
        if entry["kind"] == "dict":
            assert entry["entries"] == 0


def test_cache_info_and_clear_lru_cache():
    """Test cache_info and clear_cache with lru_cache-wrapped functions."""
    from functools import lru_cache

    class FakeBase:
        _CACHE_CATEGORIES = {
            "base": ("_cache_releases",),
        }

        def __init__(self):
            @lru_cache(maxsize=32)
            def _releases_fn(key):
                return key

            self._cache_releases = _releases_fn

        _iter_cache_attrs = AnophelesBase._iter_cache_attrs
        _estimate_cache_entry_nbytes = staticmethod(
            AnophelesBase._estimate_cache_entry_nbytes
        )
        cache_info = AnophelesBase.cache_info
        clear_cache = AnophelesBase.clear_cache

    fake = FakeBase()

    # Populate the lru_cache.
    fake._cache_releases("a")
    fake._cache_releases("b")

    info = fake.cache_info()
    assert "_cache_releases" in info
    assert info["_cache_releases"]["kind"] == "lru_cache"
    assert info["_cache_releases"]["entries"] == 2

    # Clear and verify.
    fake.clear_cache("base")
    info = fake.cache_info()
    assert info["_cache_releases"]["entries"] == 0


def test_cache_info_and_clear_single_value():
    """Test cache_info and clear_cache with single-value (non-dict, non-lru) caches."""

    class FakeBase:
        _CACHE_CATEGORIES = {
            "genome_sequence": ("_cache_genome",),
        }

        def __init__(self):
            # Simulate a single-value cache (e.g. a zarr group stored directly).
            self._cache_genome = np.zeros((10, 10), dtype=np.float64)

        _iter_cache_attrs = AnophelesBase._iter_cache_attrs
        _estimate_cache_entry_nbytes = staticmethod(
            AnophelesBase._estimate_cache_entry_nbytes
        )
        cache_info = AnophelesBase.cache_info
        clear_cache = AnophelesBase.clear_cache

    fake = FakeBase()

    info = fake.cache_info()
    assert "_cache_genome" in info
    assert info["_cache_genome"]["kind"] == "other"
    assert info["_cache_genome"]["entries"] == 1
    assert info["_cache_genome"]["nbytes"] == 800  # 10*10*8 bytes

    # Clear and verify it gets set to None.
    fake.clear_cache("genome_sequence")
    assert fake._cache_genome is None

    # After clearing, cache_info should not list it (it's None now).
    info = fake.cache_info()
    assert "_cache_genome" not in info
