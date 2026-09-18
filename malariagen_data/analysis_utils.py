import pandas as pd


def summarize_samples_by_species_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize mosquito samples by species and country.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing mosquito sample metadata with
        columns 'species' and 'country'.

    Returns
    -------
    pandas.DataFrame
        Summary table with sample counts.
    """
#Used AI model for writing this part of the code, and have cross verified the code behaviour before raising PR.

    if "species" not in df.columns or "country" not in df.columns:
        raise ValueError("DataFrame must contain 'species' and 'country' columns")

    summary = (
        df.groupby(["species", "country"])
        .size()
        .reset_index(name="sample_count")
        .sort_values("sample_count", ascending=False)
    )

    return summary