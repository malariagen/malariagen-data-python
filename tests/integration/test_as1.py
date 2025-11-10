from malariagen_data import As1


def setup_as1(url="simplecache::gs://vo_aste_release_master_us_central1/", **kwargs):
    kwargs.setdefault("check_location", False)
    kwargs.setdefault("show_progress", False)
    if url is None:
        # Test the default URL.
        # Note: This only tests the setup_as1 default URL, not the As1 default.
        # The test_anopheles setup_subclass tests the true defaults.
        return As1(**kwargs)
    if url.startswith("simplecache::"):
        # Configure the directory on the local file system to cache data.
        kwargs["simplecache"] = dict(cache_storage="gcs_cache")
    return As1(url, **kwargs)


def test_repr():
    as1 = setup_as1(check_location=True)
    assert isinstance(as1, As1)
    r = repr(as1)
    assert isinstance(r, str)
