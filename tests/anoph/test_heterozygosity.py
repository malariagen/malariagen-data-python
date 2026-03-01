import numpy as np
import pytest
import malariagen_data
from malariagen_data.anopheles import AnophelesDataResource  
from malariagen_data.util import Region

@pytest.fixture
def fake_windows_counts():
    # pretend we have two windows
    windows = np.array([[0, 10], [10, 20]])
    counts  = np.array([3, 7])
    return windows, counts

def test_heterozygosity_wraps_sample_count_het(monkeypatch, fake_windows_counts):
    # Define a dummy logger with a debug method
    class DummyLogger:
        def debug(self, *args, **kwargs):
            pass

    # monkey-patch __init__ to set up a dummy _log attribute
    monkeypatch.setattr(
        AnophelesDataResource,
        "__init__",
        lambda self, *args, **kwargs: setattr(self, "_log", DummyLogger())
    )

    # monkey-patch the private helper to return (sid, sset, windows, counts)
    def fake_sample_count_het(self, sample, region, site_mask, window_size, sample_set, chunks, inline_array):
        return "S1", "setA", fake_windows_counts[0], fake_windows_counts[1]

    monkeypatch.setattr(AnophelesDataResource, "_sample_count_het", fake_sample_count_het)

    resource = AnophelesDataResource()  
    # call for public method
    windows, counts = resource.heterozygosity(
        sample="any_sample",
        region=Region(contig="2L", start=100, end=200),
        window_size=10,
    )

    # assert that we got exactly the arrays the fake helper returned
    assert np.array_equal(windows, fake_windows_counts[0])
    assert np.array_equal(counts,  fake_windows_counts[1])