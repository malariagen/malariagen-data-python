import pytest

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.sample_metadata import AnophelesSampleMetadata


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesSampleMetadata(
        url=ag3_sim_fixture.url,
        config_path=_ag3.CONFIG_PATH,
        gcs_url=_ag3.GCS_URL,
        major_version_number=_ag3.MAJOR_VERSION_NUMBER,
        major_version_path=_ag3.MAJOR_VERSION_PATH,
        pre=True,
        aim_metadata_columns=[
            "aim_species_fraction_arab",
            "aim_species_fraction_colu",
            "aim_species_fraction_colu_no2l",
            "aim_species_gambcolu_arabiensis",
            "aim_species_gambiae_coluzzii",
            "aim_species",
        ],
        aim_metadata_dtype={
            "aim_species_fraction_arab": "float64",
            "aim_species_fraction_colu": "float64",
            "aim_species_fraction_colu_no2l": "float64",
            "aim_species_gambcolu_arabiensis": object,
            "aim_species_gambiae_coluzzii": object,
            "aim_species": object,
        },
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesSampleMetadata(
        url=af1_sim_fixture.url,
        config_path=_af1.CONFIG_PATH,
        gcs_url=_af1.GCS_URL,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
    )


def case_ag3_sim(ag3_sim_fixture, ag3_sim_api):
    return ag3_sim_fixture, ag3_sim_api


def case_af1_sim(af1_sim_fixture, af1_sim_api):
    return af1_sim_fixture, af1_sim_api


def general_metadata_expected_columns():
    return {
        "sample_id": "O",
        "partner_sample_id": "O",
        "contributor": "O",
        "country": "O",
        "location": "O",
        "year": "i",
        "month": "i",
        "latitude": "f",
        "longitude": "f",
        "sex_call": "O",
        "sample_set": "O",
        "release": "O",
        "quarter": "i",
    }


def validate_metadata(df, expected_columns):
    # Check column names.
    expected_column_names = list(expected_columns.keys())
    assert df.columns.to_list() == expected_column_names

    # Check column types.
    for c in df.columns:
        assert df[c].dtype.kind == expected_columns[c]


@pytest.mark.parametrize(
    "sample_set", ["AG1000G-AO", "AG1000G-BF-A", "1177-VO-ML-LEHMANN-VMF00004"]
)
def test_general_metadata__ag3_single_sample_set(ag3_sim_api, sample_set):
    df = ag3_sim_api.general_metadata(sample_sets=sample_set)
    validate_metadata(df, general_metadata_expected_columns())

    # Check number of rows.
    sample_count = ag3_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sample_count.loc[sample_set]
    assert len(df) == expected_len


@pytest.mark.parametrize(
    "sample_set",
    [
        "1229-VO-GH-DADZIE-VMF00095",
        "1230-VO-GA-CF-AYALA-VMF00045",
        "1231-VO-MULTI-WONDJI-VMF00043",
    ],
)
def test_general_metadata__af1_single_sample_set(af1_sim_api, sample_set):
    df = af1_sim_api.general_metadata(sample_sets=sample_set)
    validate_metadata(df, general_metadata_expected_columns())

    # Check number of rows.
    sample_count = af1_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sample_count.loc[sample_set]
    assert len(df) == expected_len


def test_general_metadata__ag3_multiple_sample_sets(ag3_sim_api):
    sample_sets = ["AG1000G-AO", "1177-VO-ML-LEHMANN-VMF00004"]
    df = ag3_sim_api.general_metadata(sample_sets=sample_sets)
    validate_metadata(df, general_metadata_expected_columns())

    # Check number of rows.
    sample_count = ag3_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sum([sample_count.loc[s] for s in sample_sets])
    assert len(df) == expected_len


def test_general_metadata__af1_multiple_sample_sets(af1_sim_api):
    sample_sets = ["1230-VO-GA-CF-AYALA-VMF00045", "1231-VO-MULTI-WONDJI-VMF00043"]
    df = af1_sim_api.general_metadata(sample_sets=sample_sets)
    validate_metadata(df, general_metadata_expected_columns())

    # Check number of rows.
    sample_count = af1_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sum([sample_count.loc[s] for s in sample_sets])
    assert len(df) == expected_len


def test_general_metadata__ag3_release(ag3_sim_api):
    release = "3.0"
    df = ag3_sim_api.general_metadata(sample_sets=release)
    validate_metadata(df, general_metadata_expected_columns())

    # Check number of rows.
    expected_len = ag3_sim_api.sample_sets(release=release)["sample_count"].sum()
    assert len(df) == expected_len


def test_general_metadata__af1_release(af1_sim_api):
    release = "1.0"
    df = af1_sim_api.general_metadata(sample_sets=release)
    validate_metadata(df, general_metadata_expected_columns())

    # Check number of rows.
    expected_len = af1_sim_api.sample_sets(release=release)["sample_count"].sum()
    assert len(df) == expected_len


def aim_metadata_expected_columns():
    return {
        "sample_id": "O",
        "aim_species_fraction_arab": "f",
        "aim_species_fraction_colu": "f",
        "aim_species_fraction_colu_no2l": "f",
        "aim_species_gambcolu_arabiensis": "O",
        "aim_species_gambiae_coluzzii": "O",
        "aim_species": "O",
    }


def validate_aim_metadata(df):
    validate_metadata(df, aim_metadata_expected_columns())

    # Check some values.
    expected_species = {
        "gambiae",
        "coluzzii",
        "arabiensis",
        "intermediate_gambcolu_arabiensis",
        "intermediate_gambiae_coluzzii",
    }
    for v in df["aim_species"]:
        assert v in expected_species


@pytest.mark.parametrize(
    "sample_set", ["AG1000G-AO", "AG1000G-BF-A", "1177-VO-ML-LEHMANN-VMF00004"]
)
def test_aim_metadata__ag3_single_sample_set(ag3_sim_api, sample_set):
    df = ag3_sim_api.aim_metadata(sample_sets=sample_set)
    validate_aim_metadata(df)

    # Check number of rows.
    sample_count = ag3_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sample_count.loc[sample_set]
    assert len(df) == expected_len


def test_aim_metadata__ag3_multiple_sample_sets(ag3_sim_api):
    sample_sets = ["AG1000G-AO", "1177-VO-ML-LEHMANN-VMF00004"]
    df = ag3_sim_api.aim_metadata(sample_sets=sample_sets)
    validate_aim_metadata(df)

    # Check number of rows.
    sample_count = ag3_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sum([sample_count.loc[s] for s in sample_sets])
    assert len(df) == expected_len


def test_aim_metadata__ag3_release(ag3_sim_api):
    release = "3.0"
    df = ag3_sim_api.aim_metadata(sample_sets=release)
    validate_aim_metadata(df)

    # Check number of rows.
    expected_len = ag3_sim_api.sample_sets(release=release)["sample_count"].sum()
    assert len(df) == expected_len


def cohorts_metadata_expected_columns(has_quarter):
    if has_quarter:
        return {
            "sample_id": "O",
            "country_iso": "O",
            "admin1_name": "O",
            "admin1_iso": "O",
            "admin2_name": "O",
            "taxon": "O",
            "cohort_admin1_year": "O",
            "cohort_admin1_month": "O",
            "cohort_admin1_quarter": "O",
            "cohort_admin2_year": "O",
            "cohort_admin2_month": "O",
            "cohort_admin2_quarter": "O",
        }
    else:
        return {
            "sample_id": "O",
            "country_iso": "O",
            "admin1_name": "O",
            "admin1_iso": "O",
            "admin2_name": "O",
            "taxon": "O",
            "cohort_admin1_year": "O",
            "cohort_admin1_month": "O",
            "cohort_admin2_year": "O",
            "cohort_admin2_month": "O",
        }


def validate_cohorts_metadata(df, has_quarter):
    # N.B., older cohorts metadata only has cohorts by year and month.
    # Newer cohorts metadata also has cohorts by quarter.
    expected_columns = cohorts_metadata_expected_columns(has_quarter=has_quarter)
    validate_metadata(df, expected_columns)


@pytest.mark.parametrize(
    "sample_set", ["AG1000G-AO", "AG1000G-BF-A", "1177-VO-ML-LEHMANN-VMF00004"]
)
def test_cohorts_metadata__ag3_single_sample_set(ag3_sim_api, sample_set):
    df = ag3_sim_api.cohorts_metadata(sample_sets=sample_set)
    validate_cohorts_metadata(df, has_quarter=True)

    # Check number of rows.
    sample_count = ag3_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sample_count.loc[sample_set]
    assert len(df) == expected_len


@pytest.mark.parametrize(
    "sample_set",
    [
        "1229-VO-GH-DADZIE-VMF00095",
        "1230-VO-GA-CF-AYALA-VMF00045",
        "1231-VO-MULTI-WONDJI-VMF00043",
    ],
)
def test_cohorts_metadata__af1_single_sample_set(af1_sim_api, sample_set):
    df = af1_sim_api.cohorts_metadata(sample_sets=sample_set)
    validate_cohorts_metadata(df, has_quarter=False)

    # Check number of rows.
    sample_count = af1_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sample_count.loc[sample_set]
    assert len(df) == expected_len


def test_cohorts_metadata__ag3_multiple_sample_sets(ag3_sim_api):
    sample_sets = ["AG1000G-AO", "1177-VO-ML-LEHMANN-VMF00004"]
    df = ag3_sim_api.cohorts_metadata(sample_sets=sample_sets)
    validate_cohorts_metadata(df, has_quarter=True)

    # Check number of rows.
    sample_count = ag3_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sum([sample_count.loc[s] for s in sample_sets])
    assert len(df) == expected_len


def test_cohorts_metadata__af1_multiple_sample_sets(af1_sim_api):
    sample_sets = ["1230-VO-GA-CF-AYALA-VMF00045", "1231-VO-MULTI-WONDJI-VMF00043"]
    df = af1_sim_api.cohorts_metadata(sample_sets=sample_sets)
    validate_cohorts_metadata(df, has_quarter=False)

    # Check number of rows.
    sample_count = af1_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sum([sample_count.loc[s] for s in sample_sets])
    assert len(df) == expected_len


def test_cohorts_metadata__ag3_release(ag3_sim_api):
    release = "3.0"
    df = ag3_sim_api.cohorts_metadata(sample_sets=release)
    validate_cohorts_metadata(df, has_quarter=True)

    # Check number of rows.
    expected_len = ag3_sim_api.sample_sets(release=release)["sample_count"].sum()
    assert len(df) == expected_len


def test_cohorts_metadata__af1_release(af1_sim_api):
    release = "1.0"
    df = af1_sim_api.cohorts_metadata(sample_sets=release)
    validate_cohorts_metadata(df, has_quarter=False)

    # Check number of rows.
    expected_len = af1_sim_api.sample_sets(release=release)["sample_count"].sum()
    assert len(df) == expected_len


def sample_metadata_expected_columns(has_aims, has_cohorts_by_quarter):
    expected_columns = general_metadata_expected_columns()
    if has_aims:
        expected_columns.update(aim_metadata_expected_columns())
    expected_columns.update(
        cohorts_metadata_expected_columns(has_quarter=has_cohorts_by_quarter)
    )
    return expected_columns


@pytest.mark.parametrize(
    "sample_set", ["AG1000G-AO", "AG1000G-BF-A", "1177-VO-ML-LEHMANN-VMF00004"]
)
def test_sample_metadata__ag3_single_sample_set(ag3_sim_api, sample_set):
    df = ag3_sim_api.sample_metadata(sample_sets=sample_set)
    validate_metadata(
        df, sample_metadata_expected_columns(has_aims=True, has_cohorts_by_quarter=True)
    )

    # Check number of rows.
    sample_count = ag3_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sample_count.loc[sample_set]
    assert len(df) == expected_len


@pytest.mark.parametrize(
    "sample_set",
    [
        "1229-VO-GH-DADZIE-VMF00095",
        "1230-VO-GA-CF-AYALA-VMF00045",
        "1231-VO-MULTI-WONDJI-VMF00043",
    ],
)
def test_sample_metadata__af1_single_sample_set(af1_sim_api, sample_set):
    df = af1_sim_api.sample_metadata(sample_sets=sample_set)
    validate_metadata(
        df,
        sample_metadata_expected_columns(has_aims=False, has_cohorts_by_quarter=False),
    )

    # Check number of rows.
    sample_count = af1_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sample_count.loc[sample_set]
    assert len(df) == expected_len


def test_sample_metadata__ag3_multiple_sample_sets(ag3_sim_api):
    sample_sets = ["AG1000G-AO", "1177-VO-ML-LEHMANN-VMF00004"]
    df = ag3_sim_api.sample_metadata(sample_sets=sample_sets)
    validate_metadata(
        df, sample_metadata_expected_columns(has_aims=True, has_cohorts_by_quarter=True)
    )

    # Check number of rows.
    sample_count = ag3_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sum([sample_count.loc[s] for s in sample_sets])
    assert len(df) == expected_len


def test_sample_metadata__af1_multiple_sample_sets(af1_sim_api):
    sample_sets = ["1230-VO-GA-CF-AYALA-VMF00045", "1231-VO-MULTI-WONDJI-VMF00043"]
    df = af1_sim_api.sample_metadata(sample_sets=sample_sets)
    validate_metadata(
        df,
        sample_metadata_expected_columns(has_aims=False, has_cohorts_by_quarter=False),
    )

    # Check number of rows.
    sample_count = af1_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sum([sample_count.loc[s] for s in sample_sets])
    assert len(df) == expected_len


def test_sample_metadata__ag3_release(ag3_sim_api):
    release = "3.0"
    df = ag3_sim_api.sample_metadata(sample_sets=release)
    validate_metadata(
        df, sample_metadata_expected_columns(has_aims=True, has_cohorts_by_quarter=True)
    )

    # Check number of rows.
    expected_len = ag3_sim_api.sample_sets(release=release)["sample_count"].sum()
    assert len(df) == expected_len


def test_sample_metadata__af1_release(af1_sim_api):
    release = "1.0"
    df = af1_sim_api.sample_metadata(sample_sets=release)
    validate_metadata(
        df,
        sample_metadata_expected_columns(has_aims=False, has_cohorts_by_quarter=False),
    )

    # Check number of rows.
    expected_len = af1_sim_api.sample_sets(release=release)["sample_count"].sum()
    assert len(df) == expected_len


# TODO Test extra metadata.
