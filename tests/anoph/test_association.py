import malariagen_data
import pandas as pd
import plotly.graph_objects as go  # type: ignore
import pytest
import xarray as xr


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return malariagen_data.Ag3(
        url=ag3_sim_fixture.url,
        public_url=ag3_sim_fixture.url,
        pre=True,
        check_location=False,
        show_progress=False,
        bokeh_output_notebook=False,
        results_cache=ag3_sim_fixture.results_cache_path.as_posix(),
    )


def test_snp_phenotype_association_produces_plot_input(ag3_sim_api):
    df = ag3_sim_api.snp_phenotype_association(
        region="2L",
        sample_sets=["AG1000G-AO"],
    )

    assert isinstance(df, pd.DataFrame)
    assert {"contig", "position", "pvalue"}.issubset(df.columns)
    assert len(df) > 0
    assert (df["pvalue"].dropna() >= 0).all()
    assert (df["pvalue"].dropna() <= 1).all()

    manhattan = ag3_sim_api.plot_manhattan(df, show=False)
    qq = ag3_sim_api.plot_qq(df, show=False)
    assert isinstance(manhattan, go.Figure)
    assert isinstance(qq, go.Figure)


def test_plot_association_from_xarray(ag3_sim_api):
    ds = xr.Dataset(
        {
            "contig": ("variants", ["2L", "2L", "2R", "2R"]),
            "position": ("variants", [100, 200, 100, 250]),
            "pvalue": ("variants", [0.5, 1e-6, 0.1, 1e-9]),
        }
    )

    manhattan = ag3_sim_api.plot_manhattan(ds, show=False)
    qq = ag3_sim_api.plot_qq(ds, show=False)
    assert isinstance(manhattan, go.Figure)
    assert isinstance(qq, go.Figure)


def test_plot_manhattan_missing_column_raises(ag3_sim_api):
    data = pd.DataFrame({"contig": ["2L"], "position": [100]})
    with pytest.raises(ValueError, match="Missing required columns"):
        ag3_sim_api.plot_manhattan(data, show=False)
