import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data import adir1 as _adir1
from malariagen_data.anoph.base import AnophelesBase


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesBase(
        url=ag3_sim_fixture.url,
        public_url=ag3_sim_fixture.url,
        config_path=_ag3.CONFIG_PATH,
        major_version_number=_ag3.MAJOR_VERSION_NUMBER,
        major_version_path=_ag3.MAJOR_VERSION_PATH,
        pre=True,
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesBase(
        url=af1_sim_fixture.url,
        public_url=af1_sim_fixture.url,
        config_path=_af1.CONFIG_PATH,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
    )


@pytest.fixture
def adir1_sim_api(adir1_sim_fixture):
    return AnophelesBase(
        url=adir1_sim_fixture.url,
        public_url=adir1_sim_fixture.url,
        config_path=_adir1.CONFIG_PATH,
        major_version_number=_adir1.MAJOR_VERSION_NUMBER,
        major_version_path=_adir1.MAJOR_VERSION_PATH,
        pre=False,
    )


@pytest.fixture
def amin1_sim_api(amin1_sim_fixture):
    return AnophelesBase(
        url=amin1_sim_fixture.url,
        public_url=amin1_sim_fixture.url,
        config_path=_adir1.CONFIG_PATH,
        major_version_number=_adir1.MAJOR_VERSION_NUMBER,
        major_version_path=_adir1.MAJOR_VERSION_PATH,
        pre=False,
    )


# N.B., here we use pytest_cases to parametrize tests. Each
# function whose name begins with "case_" defines a set of
# inputs to the test functions. See the documentation for
# pytest_cases for more information, e.g.:
#
# https://smarie.github.io/python-pytest-cases/#basic-usage
#
# We use this approach here because we want to use fixtures
# as test parameters, which is otherwise hard to do with
# pytest alone.


def case_ag3_sim(ag3_sim_fixture, ag3_sim_api):
    return ag3_sim_fixture, ag3_sim_api


def case_af1_sim(af1_sim_fixture, af1_sim_api):
    return af1_sim_fixture, af1_sim_api


def case_adir1_sim(adir1_sim_fixture, adir1_sim_api):
    return adir1_sim_fixture, adir1_sim_api


def case_amin1_sim(amin1_sim_fixture, amin1_sim_api):
    return amin1_sim_fixture, amin1_sim_api


@parametrize_with_cases("fixture,api", cases=".")
def test_config(fixture, api):
    config = api.config
    assert isinstance(config, dict)
    assert config == fixture.config


@parametrize_with_cases("fixture,api", cases=".")
def test_releases(fixture, api):
    releases = api.releases
    assert isinstance(releases, tuple)
    assert len(releases) > 0
    assert all([isinstance(r, str) for r in releases])
    assert releases == fixture.releases


@parametrize_with_cases("fixture,api", cases=".")
def test_client_location(fixture, api):
    location = api.client_location
    assert isinstance(location, str)


@parametrize_with_cases("fixture,api", cases=".")
def test_sample_sets_default(fixture, api):
    df = api.sample_sets()
    releases = api.releases
    expected = pd.concat(
        [fixture.release_manifests[release] for release in releases],
        axis=0,
        ignore_index=True,
    )
    expected.reset_index(inplace=True, drop=True)
    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == [
        "sample_set",
        "sample_count",
        "study_id",
        "study_url",
        "terms_of_use_expiry_date",
        "terms_of_use_url",
        "release",
        "unrestricted_use",
    ]
    assert len(df) > 0
    assert_frame_equal(
        df[
            [
                "sample_set",
                "sample_count",
                "study_id",
                "study_url",
                "terms_of_use_expiry_date",
                "terms_of_use_url",
            ]
        ],
        expected,
    )


@parametrize_with_cases("fixture,api", cases=".")
def test_sample_sets_release(fixture, api):
    releases = api.releases
    for release in releases:
        df_ss = api.sample_sets(release=release)
        assert isinstance(df_ss, pd.DataFrame)
        assert df_ss.columns.tolist() == [
            "sample_set",
            "sample_count",
            "study_id",
            "study_url",
            "terms_of_use_expiry_date",
            "terms_of_use_url",
            "release",
            "unrestricted_use",
        ]
        assert len(df_ss) > 0
        expected = fixture.release_manifests[release]
        assert_frame_equal(
            df_ss[
                [
                    "sample_set",
                    "sample_count",
                    "study_id",
                    "study_url",
                    "terms_of_use_expiry_date",
                    "terms_of_use_url",
                ]
            ],
            expected,
        )
        assert (df_ss["release"] == release).all()

    with pytest.raises(TypeError):
        api.sample_sets(release=3.1)  # type: ignore


@parametrize_with_cases("fixture,api", cases=".")
def test_lookup_release(fixture, api):
    releases = api.releases
    for release in releases:
        df_ss = api.sample_sets(release=release)
        for s in df_ss["sample_set"]:
            assert api.lookup_release(s) == release


def test_prep_sample_sets_param(ag3_sim_api: AnophelesBase):
    assert ag3_sim_api._prep_sample_sets_param(sample_sets="3.0") == [
        "AG1000G-AO",
        "AG1000G-BF-A",
    ]
    assert ag3_sim_api._prep_sample_sets_param(sample_sets="3.1") == [
        "1177-VO-ML-LEHMANN-VMF00004",
    ]
    assert ag3_sim_api._prep_sample_sets_param(sample_sets=["3.0", "3.1"]) == [
        "1177-VO-ML-LEHMANN-VMF00004",
        "AG1000G-AO",
        "AG1000G-BF-A",
    ]
    assert ag3_sim_api._prep_sample_sets_param(sample_sets=None) == [
        "1177-VO-ML-LEHMANN-VMF00004",
        "AG1000G-AO",
        "AG1000G-BF-A",
    ]
    assert ag3_sim_api._prep_sample_sets_param(sample_sets="AG1000G-AO") == [
        "AG1000G-AO"
    ]
    assert ag3_sim_api._prep_sample_sets_param(
        sample_sets=["AG1000G-AO", "AG1000G-BF-A"]
    ) == [
        "AG1000G-AO",
        "AG1000G-BF-A",
    ]
    assert ag3_sim_api._prep_sample_sets_param(
        sample_sets=("AG1000G-AO", "AG1000G-BF-A")
    ) == [
        "AG1000G-AO",
        "AG1000G-BF-A",
    ]
    assert ag3_sim_api._prep_sample_sets_param(
        sample_sets=["AG1000G-AO", "AG1000G-BF-A", "AG1000G-AO"]
    ) == [
        "AG1000G-AO",
        "AG1000G-BF-A",
    ]
    assert ag3_sim_api._prep_sample_sets_param(sample_sets=["3.0", "AG1000G-AO"]) == [
        "AG1000G-AO",
        "AG1000G-BF-A",
    ]
    with pytest.raises(ValueError):
        ag3_sim_api._prep_sample_sets_param(sample_sets=["AG1000G-AO", "foobar"])


@parametrize_with_cases("fixture,api", cases=".")
def test_lookup_study(fixture, api):
    # Set up test.
    df_sample_sets = api.sample_sets()
    all_sample_sets = df_sample_sets["sample_set"].values
    sample_set = np.random.choice(all_sample_sets)

    study_rec_by_sample_set = api.lookup_study(sample_set)
    df_sample_set = df_sample_sets.set_index("sample_set").loc[sample_set]

    # Check we get the same study_id back.
    assert df_sample_set["study_id"] == study_rec_by_sample_set

    # Check we get a study_id string.
    assert isinstance(study_rec_by_sample_set, str)

    with pytest.raises(ValueError):
        api.lookup_study("foobar")


def _strip_terms_of_use_from_manifest(manifest_path):
    """Rewrite a manifest TSV file without terms-of-use columns."""
    df = pd.read_csv(manifest_path, sep="\t")
    cols_to_drop = [c for c in df.columns if c.startswith("terms_of_use")]
    df = df.drop(columns=cols_to_drop)
    df.to_csv(manifest_path, index=False, sep="\t")


def test_lookup_terms_of_use_info_missing_columns(ag3_sim_fixture):
    import shutil

    manifest_paths = [
        ag3_sim_fixture.bucket_path / "v3" / "manifest.tsv",
        ag3_sim_fixture.bucket_path / "v3.1" / "manifest.tsv",
    ]
    backups = []
    for mp in manifest_paths:
        bp = mp.parent / "manifest.tsv.bak"
        shutil.copy2(mp, bp)
        backups.append(bp)

    try:
        for mp in manifest_paths:
            _strip_terms_of_use_from_manifest(mp)

        api = AnophelesBase(
            url=ag3_sim_fixture.url,
            public_url=ag3_sim_fixture.url,
            config_path=_ag3.CONFIG_PATH,
            major_version_number=_ag3.MAJOR_VERSION_NUMBER,
            major_version_path=_ag3.MAJOR_VERSION_PATH,
            pre=True,
        )

        sample_set = "1177-VO-ML-LEHMANN-VMF00004"
        with pytest.raises(ValueError, match="Terms-of-use columns missing"):
            api.lookup_terms_of_use_info(sample_set)
    finally:
        for mp, bp in zip(manifest_paths, backups):
            shutil.move(bp, mp)


def test_sample_set_has_unrestricted_use_missing_column(ag3_sim_fixture):
    import shutil

    manifest_paths = [
        ag3_sim_fixture.bucket_path / "v3" / "manifest.tsv",
        ag3_sim_fixture.bucket_path / "v3.1" / "manifest.tsv",
    ]
    backups = []
    for mp in manifest_paths:
        bp = mp.parent / "manifest.tsv.bak"
        shutil.copy2(mp, bp)
        backups.append(bp)

    try:
        for mp in manifest_paths:
            _strip_terms_of_use_from_manifest(mp)

        api = AnophelesBase(
            url=ag3_sim_fixture.url,
            public_url=ag3_sim_fixture.url,
            config_path=_ag3.CONFIG_PATH,
            major_version_number=_ag3.MAJOR_VERSION_NUMBER,
            major_version_path=_ag3.MAJOR_VERSION_PATH,
            pre=True,
        )

        sample_set = "1177-VO-ML-LEHMANN-VMF00004"
        with pytest.raises(ValueError, match="unrestricted_use.*missing"):
            api._sample_set_has_unrestricted_use(sample_set=sample_set)
    finally:
        for mp, bp in zip(manifest_paths, backups):
            shutil.move(bp, mp)


def test_sample_sets_no_terms_of_use(ag3_sim_fixture):
    import shutil

    manifest_paths = [
        ag3_sim_fixture.bucket_path / "v3" / "manifest.tsv",
        ag3_sim_fixture.bucket_path / "v3.1" / "manifest.tsv",
    ]
    backups = []
    for mp in manifest_paths:
        bp = mp.parent / "manifest.tsv.bak"
        shutil.copy2(mp, bp)
        backups.append(bp)

    try:
        for mp in manifest_paths:
            _strip_terms_of_use_from_manifest(mp)

        api = AnophelesBase(
            url=ag3_sim_fixture.url,
            public_url=ag3_sim_fixture.url,
            config_path=_ag3.CONFIG_PATH,
            major_version_number=_ag3.MAJOR_VERSION_NUMBER,
            major_version_path=_ag3.MAJOR_VERSION_PATH,
            pre=True,
        )

        df = api.sample_sets(release="3.1")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    finally:
        for mp, bp in zip(manifest_paths, backups):
            shutil.move(bp, mp)
