import pandas as pd
import plotly.graph_objects as go  # type: ignore
import pytest
import xarray as xr

from malariagen_data.anoph.association import AnophelesAssociationPlotting


class DummyAssociationApi(AnophelesAssociationPlotting):
    def phenotypes_with_snps(self, **kwargs):  # noqa: ARG002
        return xr.Dataset(
            {
                "call_genotype": (
                    ("variants", "samples", "ploidy"),
                    [
                        [[0, 0], [0, 1], [1, 1], [0, 1]],
                        [[0, 0], [0, 0], [0, 1], [1, 1]],
                        [[0, 1], [0, 1], [1, 1], [1, 1]],
                    ],
                ),
                "phenotype_binary": (("samples",), [0.0, 0.0, 1.0, 1.0]),
                "variant_contig": (("variants",), ["2L", "2L", "2R"]),
                "variant_position": (("variants",), [100, 200, 300]),
            }
        )


def test_snp_phenotype_association_produces_plot_input():
    api = DummyAssociationApi()

    df = api.snp_phenotype_association(region="2L:1-1000")

    assert isinstance(df, pd.DataFrame)
    assert {"contig", "position", "pvalue"}.issubset(df.columns)
    assert len(df) > 0
    assert (df["pvalue"].dropna() >= 0).all()
    assert (df["pvalue"].dropna() <= 1).all()

    manhattan = api.plot_manhattan(df, show=False)
    qq = api.plot_qq(df, show=False)
    assert isinstance(manhattan, go.Figure)
    assert isinstance(qq, go.Figure)


def test_plot_association_from_xarray():
    api = DummyAssociationApi()

    ds = xr.Dataset(
        {
            "contig": ("variants", ["2L", "2L", "2R", "2R"]),
            "position": ("variants", [100, 200, 100, 250]),
            "pvalue": ("variants", [0.5, 1e-6, 0.1, 1e-9]),
        }
    )

    manhattan = api.plot_manhattan(ds, show=False)
    qq = api.plot_qq(ds, show=False)
    assert isinstance(manhattan, go.Figure)
    assert isinstance(qq, go.Figure)


def test_plot_manhattan_missing_column_raises():
    api = DummyAssociationApi()
    data = pd.DataFrame({"contig": ["2L"], "position": [100]})
    with pytest.raises(ValueError, match="Missing required columns"):
        api.plot_manhattan(data, show=False)
