"""Test cache name resolver methods for MRO edge case handling."""

from malariagen_data.anopheles import AnophelesDataResource


class DummyAnophelesNoOverrides(AnophelesDataResource):
    """Dummy subclass that doesn't override cache name properties.

    Tests the edge case where resolver methods must return defaults
    instead of raising NotImplementedError.
    """

    def __init__(self):
        # Minimal initialization - we only test the resolver methods
        pass


def test_get_xpehh_gwss_cache_name_returns_default():
    """Test that _get_xpehh_gwss_cache_name returns default when not overridden."""
    instance = DummyAnophelesNoOverrides()

    # Should not raise NotImplementedError
    cache_name = instance._get_xpehh_gwss_cache_name()

    # Should return default string
    assert isinstance(cache_name, str)
    assert cache_name == "xpehh_gwss_v1"


def test_get_ihs_gwss_cache_name_returns_default():
    """Test that _get_ihs_gwss_cache_name returns default when not overridden."""
    instance = DummyAnophelesNoOverrides()

    # Should not raise NotImplementedError
    cache_name = instance._get_ihs_gwss_cache_name()

    # Should return default string
    assert isinstance(cache_name, str)
    assert cache_name == "ihs_gwss_v1"


def test_cache_name_resolvers_always_return_string():
    """Test that both resolvers always return strings, never other types."""
    instance = DummyAnophelesNoOverrides()

    xpehh_name = instance._get_xpehh_gwss_cache_name()
    ihs_name = instance._get_ihs_gwss_cache_name()

    assert isinstance(xpehh_name, str), f"Expected str, got {type(xpehh_name).__name__}"
    assert isinstance(ihs_name, str), f"Expected str, got {type(ihs_name).__name__}"
