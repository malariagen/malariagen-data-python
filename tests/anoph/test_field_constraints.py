"""Tests for pydantic Field() numeric constraints on parameter type aliases.

These tests verify that the Field() constraints added to *_params.py files
are enforced at runtime by the _check_types decorator (which uses pydantic
validate_call under the hood).
"""

import pytest

from malariagen_data.util import _check_types


# ---------------------------------------------------------------------------
# Helper: a minimal function decorated with @_check_types that exercises
# various constrained parameter types from the *_params.py modules.
# ---------------------------------------------------------------------------


@_check_types
def _fn_cohort_size(
    cohort_size: int,
) -> int:
    """Uses base_params.cohort_size (Field(ge=1))."""
    # Import the type alias so pydantic sees the Annotated metadata.
    from malariagen_data.anoph.base_params import cohort_size as cohort_size_type  # noqa: F401

    return cohort_size


# Because _check_types reads the type hints from the function signature at
# decoration time, and base_params types are TypeAliases (not actual
# Annotated types when used bare as `int`), we need to use the TypeAlias
# directly in the signature. Let's create properly-typed helper functions.


from malariagen_data.anoph import base_params, het_params, ihs_params  # noqa: E402


@_check_types
def _fn_with_confidence_level(
    confidence_level: base_params.confidence_level,
) -> float:
    return confidence_level


@_check_types
def _fn_with_n_snps(
    n_snps: base_params.n_snps,
) -> int:
    return n_snps


@_check_types
def _fn_with_cohort_size(
    cohort_size: base_params.cohort_size,
) -> int:
    return cohort_size


@_check_types
def _fn_with_min_cohort_size(
    min_cohort_size: base_params.min_cohort_size,
) -> int:
    return min_cohort_size


@_check_types
def _fn_with_random_seed(
    random_seed: base_params.random_seed,
) -> int:
    return random_seed


@_check_types
def _fn_with_thin_offset(
    thin_offset: base_params.thin_offset,
) -> int:
    return thin_offset


@_check_types
def _fn_with_n_jack(
    n_jack: base_params.n_jack,
) -> int:
    return n_jack


@_check_types
def _fn_with_phet_roh(
    phet_roh: het_params.phet_roh,
) -> float:
    return phet_roh


@_check_types
def _fn_with_filter_min_maf(
    filter_min_maf: ihs_params.filter_min_maf,
) -> float:
    return filter_min_maf


# ---------------------------------------------------------------------------
# Tests: valid values should pass
# ---------------------------------------------------------------------------


class TestFieldConstraintsValidValues:
    """Verify that valid values are accepted without error."""

    def test_confidence_level_valid(self):
        assert _fn_with_confidence_level(confidence_level=0.95) == 0.95
        assert _fn_with_confidence_level(confidence_level=0.5) == 0.5
        assert _fn_with_confidence_level(confidence_level=0.01) == 0.01

    def test_n_snps_valid(self):
        assert _fn_with_n_snps(n_snps=1) == 1
        assert _fn_with_n_snps(n_snps=1000) == 1000

    def test_cohort_size_valid(self):
        assert _fn_with_cohort_size(cohort_size=1) == 1
        assert _fn_with_cohort_size(cohort_size=50) == 50

    def test_random_seed_valid(self):
        assert _fn_with_random_seed(random_seed=0) == 0
        assert _fn_with_random_seed(random_seed=42) == 42

    def test_thin_offset_valid(self):
        assert _fn_with_thin_offset(thin_offset=0) == 0
        assert _fn_with_thin_offset(thin_offset=10) == 10

    def test_phet_roh_valid(self):
        assert _fn_with_phet_roh(phet_roh=0.0) == 0.0
        assert _fn_with_phet_roh(phet_roh=0.5) == 0.5
        assert _fn_with_phet_roh(phet_roh=1.0) == 1.0

    def test_filter_min_maf_valid(self):
        assert _fn_with_filter_min_maf(filter_min_maf=0.0) == 0.0
        assert _fn_with_filter_min_maf(filter_min_maf=0.05) == 0.05
        assert _fn_with_filter_min_maf(filter_min_maf=0.5) == 0.5


# ---------------------------------------------------------------------------
# Tests: invalid values should raise TypeError (from _check_types wrapper)
# ---------------------------------------------------------------------------


class TestFieldConstraintsInvalidValues:
    """Verify that out-of-range values are rejected."""

    def test_confidence_level_zero(self):
        with pytest.raises(TypeError):
            _fn_with_confidence_level(confidence_level=0.0)

    def test_confidence_level_one(self):
        with pytest.raises(TypeError):
            _fn_with_confidence_level(confidence_level=1.0)

    def test_confidence_level_negative(self):
        with pytest.raises(TypeError):
            _fn_with_confidence_level(confidence_level=-0.5)

    def test_confidence_level_above_one(self):
        with pytest.raises(TypeError):
            _fn_with_confidence_level(confidence_level=1.5)

    def test_n_snps_zero(self):
        with pytest.raises(TypeError):
            _fn_with_n_snps(n_snps=0)

    def test_n_snps_negative(self):
        with pytest.raises(TypeError):
            _fn_with_n_snps(n_snps=-1)

    def test_cohort_size_zero(self):
        with pytest.raises(TypeError):
            _fn_with_cohort_size(cohort_size=0)

    def test_cohort_size_negative(self):
        with pytest.raises(TypeError):
            _fn_with_cohort_size(cohort_size=-5)

    def test_min_cohort_size_zero(self):
        with pytest.raises(TypeError):
            _fn_with_min_cohort_size(min_cohort_size=0)

    def test_random_seed_negative(self):
        with pytest.raises(TypeError):
            _fn_with_random_seed(random_seed=-1)

    def test_thin_offset_negative(self):
        with pytest.raises(TypeError):
            _fn_with_thin_offset(thin_offset=-1)

    def test_n_jack_zero(self):
        with pytest.raises(TypeError):
            _fn_with_n_jack(n_jack=0)

    def test_phet_roh_negative(self):
        with pytest.raises(TypeError):
            _fn_with_phet_roh(phet_roh=-0.1)

    def test_phet_roh_above_one(self):
        with pytest.raises(TypeError):
            _fn_with_phet_roh(phet_roh=1.1)

    def test_filter_min_maf_negative(self):
        with pytest.raises(TypeError):
            _fn_with_filter_min_maf(filter_min_maf=-0.01)

    def test_filter_min_maf_above_half(self):
        with pytest.raises(TypeError):
            _fn_with_filter_min_maf(filter_min_maf=0.6)
