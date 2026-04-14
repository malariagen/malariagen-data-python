from malariagen_data import Afar1


def setup_afar1(
    url="simplecache::gs://vo_afar_release_master_us_central1/", **kwargs
):
    kwargs.setdefault("check_location", False)
    kwargs.setdefault("show_progress", False)
    if url is None:
        return Afar1(**kwargs)
    if url.startswith("simplecache::"):
        kwargs["simplecache"] = dict(cache_storage="gcs_cache")
    return Afar1(url, **kwargs)


def test_repr():
    afar1 = setup_afar1(check_location=True)
    assert isinstance(afar1, Afar1)
    r = repr(afar1)
    assert isinstance(r, str)
