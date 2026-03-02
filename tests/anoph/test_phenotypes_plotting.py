import pandas as pd
import plotly.graph_objects as go
import pytest
import xarray as xr

from malariagen_data.anoph.phenotypes import AnophelesPhenotypeData


class DummyPhenotypes(AnophelesPhenotypeData):
    def __init__(self):
        pass


def test_plot_manhattan():
    api = DummyPhenotypes()
    data = pd.DataFrame(
        {
            "contig": ["2L", "2L", "2R", "2R"],
            "position": [100, 200, 100, 250],
            "pvalue": [0.5, 1e-6, 0.1, 1e-9],
        }
    )

    fig = api.plot_manhattan(data, show=False)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_plot_qq():
    api = DummyPhenotypes()
    data = pd.DataFrame({"pvalue": [0.9, 0.5, 0.1, 1e-3, 1e-6]})

    fig = api.plot_qq(data, show=False)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_plot_manhattan_missing_column():
    api = DummyPhenotypes()
    data = pd.DataFrame({"contig": ["2L"], "position": [100]})

    with pytest.raises(ValueError, match="Missing required columns"):
        api.plot_manhattan(data, show=False)


def test_plot_manhattan_from_xarray():
    api = DummyPhenotypes()
    ds = xr.Dataset(
        {
            "contig": ("variants", ["2L", "2L", "2R", "2R"]),
            "position": ("variants", [100, 200, 100, 250]),
            "pvalue": ("variants", [0.5, 1e-6, 0.1, 1e-9]),
        }
    )

    fig = api.plot_manhattan(ds, show=False)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_plot_qq_from_xarray():
    api = DummyPhenotypes()
    ds = xr.Dataset({"pvalue": ("variants", [0.9, 0.5, 0.1, 1e-3, 1e-6])})

    fig = api.plot_qq(ds, show=False)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
