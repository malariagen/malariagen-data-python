import pytest
import plotly.graph_objects as go  # type: ignore


@pytest.mark.skip
def test_plot_frequencies_heatmap(api, frq_df):
    fig = api.plot_frequencies_heatmap(frq_df, show=False, max_len=None)
    assert isinstance(fig, go.Figure)

    # Test max_len behaviour.
    with pytest.raises(ValueError):
        api.plot_frequencies_heatmap(frq_df, show=False, max_len=len(frq_df) - 1)

    # Test index parameter - if None, should use dataframe index.
    fig = api.plot_frequencies_heatmap(frq_df, show=False, index=None, max_len=None)
    # Not unique.
    with pytest.raises(ValueError):
        api.plot_frequencies_heatmap(frq_df, show=False, index="contig", max_len=None)
