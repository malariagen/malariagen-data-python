"""
Unit tests for the veff module.

These tests verify:
1. In-frame complex variant classification via _classify_inframe_complex_effect()
2. Proper handling of all amino acid change scenarios
3. No TODO string artifacts in effect annotations
"""


from malariagen_data.veff import _classify_inframe_complex_effect, VariantEffect


# Create a minimal base_effect for testing
def create_test_effect(**kwargs):
    """Helper to create a VariantEffect with defaults for testing."""
    defaults = {
        "effect": "PLACEHOLDER",
        "impact": "UNKNOWN",
        "chrom": "2L",
        "pos": 100,
        "ref": "ATG",
        "alt": "ATGC",
        "vlen": 1,
        "ref_start": 100,
        "ref_stop": 102,
        "strand": "+",
        "ref_codon": "ATG",
        "alt_codon": "ATGC",
        "codon_change": "ATG/ATGC",
        "aa_pos": 1,
        "ref_aa": "M",
        "alt_aa": "M",
        "aa_change": "M1M",
    }
    defaults.update(kwargs)
    return VariantEffect(**defaults)


class TestClassifyInframeComplexEffect:
    """Test cases for _classify_inframe_complex_effect() function."""

    def test_synonymous_complex_variant(self):
        """Synonymous complex variant returns INFRAME_COMPLEX_SYNONYMOUS with LOW impact."""
        base_effect = create_test_effect(ref_aa="MLR", alt_aa="MLR")
        result = _classify_inframe_complex_effect(
            base_effect, ref_cds_start=0, ref_aa="MLR", alt_aa="MLR"
        )

        assert result.effect == "INFRAME_COMPLEX_SYNONYMOUS"
        assert result.impact == "LOW"
        assert result.chrom == "2L"
        assert result.pos == 100

    def test_start_lost_complex_variant(self):
        """Start codon mutation at position 0 returns START_LOST with HIGH impact."""
        base_effect = create_test_effect(ref_aa="M", alt_aa="R")
        result = _classify_inframe_complex_effect(
            base_effect, ref_cds_start=0, ref_aa="M", alt_aa="R"
        )

        assert result.effect == "START_LOST"
        assert result.impact == "HIGH"

    def test_start_not_lost_when_not_position_zero(self):
        """M to different amino acid at non-zero position returns INFRAME_COMPLEX."""
        base_effect = create_test_effect(ref_aa="M", alt_aa="R")
        result = _classify_inframe_complex_effect(
            base_effect, ref_cds_start=3, ref_aa="M", alt_aa="R"
        )

        assert result.effect == "INFRAME_COMPLEX"
        assert result.impact == "MODERATE"

    def test_stop_lost_complex_variant(self):
        """Stop codon mutation returns STOP_LOST with HIGH impact."""
        base_effect = create_test_effect(ref_aa="*", alt_aa="R")
        result = _classify_inframe_complex_effect(
            base_effect, ref_cds_start=100, ref_aa="*", alt_aa="R"
        )

        assert result.effect == "STOP_LOST"
        assert result.impact == "HIGH"

    def test_stop_gained_complex_variant(self):
        """Complex variant introducing stop codon returns STOP_GAINED with HIGH impact."""
        base_effect = create_test_effect(ref_aa="R", alt_aa="*")
        result = _classify_inframe_complex_effect(
            base_effect, ref_cds_start=50, ref_aa="R", alt_aa="*"
        )

        assert result.effect == "STOP_GAINED"
        assert result.impact == "HIGH"

    def test_missense_complex_variant(self):
        """Missense complex variant returns INFRAME_COMPLEX with MODERATE impact."""
        base_effect = create_test_effect(ref_aa="MLR", alt_aa="MWR")
        result = _classify_inframe_complex_effect(
            base_effect, ref_cds_start=5, ref_aa="MLR", alt_aa="MWR"
        )

        assert result.effect == "INFRAME_COMPLEX"
        assert result.impact == "MODERATE"

    def test_no_todo_string_in_output(self):
        """All classification branches produce valid effect names (no TODO strings)."""
        test_cases = [
            ("INFRAME_COMPLEX_SYNONYMOUS", "M", "M", 0),
            ("START_LOST", "M", "R", 0),
            ("STOP_LOST", "*", "R", 50),
            ("STOP_GAINED", "R", "*", 50),
            ("INFRAME_COMPLEX", "L", "W", 5),
        ]

        for expected_effect, ref_aa, alt_aa, ref_cds_start in test_cases:
            base_effect = create_test_effect(ref_aa=ref_aa, alt_aa=alt_aa)
            result = _classify_inframe_complex_effect(
                base_effect, ref_cds_start=ref_cds_start, ref_aa=ref_aa, alt_aa=alt_aa
            )

            assert "TODO" not in result.effect
            assert result.effect == expected_effect
            assert result.impact != "UNKNOWN"
            assert result.impact in ("LOW", "MODERATE", "HIGH")

    def test_base_effect_preservation(self):
        """Non-effect fields are preserved in returned effect."""
        base_effect = create_test_effect(
            ref_aa="M", alt_aa="R", chrom="3R", pos=999, ref="ATGAAAA", alt="ATGTAAA"
        )
        result = _classify_inframe_complex_effect(
            base_effect, ref_cds_start=0, ref_aa="M", alt_aa="R"
        )

        assert result.chrom == "3R"
        assert result.pos == 999
        assert result.ref == "ATGAAAA"
        assert result.alt == "ATGTAAA"
        assert result.effect == "START_LOST"
        assert result.impact == "HIGH"

    def test_multi_codon_missense(self):
        """Multi-codon missense variant returns INFRAME_COMPLEX with MODERATE impact."""
        base_effect = create_test_effect(ref_aa="KP", alt_aa="KF")
        result = _classify_inframe_complex_effect(
            base_effect, ref_cds_start=3, ref_aa="KP", alt_aa="KF"
        )

        assert result.effect == "INFRAME_COMPLEX"
        assert result.impact == "MODERATE"


class TestInframeComplexEffectIntegration:
    """Integration tests to ensure the fix works end-to-end."""

    def test_complex_variant_effect_not_unknown(self):
        """Complex variants have valid impact (not UNKNOWN)."""
        base_effect = create_test_effect(ref_aa="L", alt_aa="W")
        result = _classify_inframe_complex_effect(
            base_effect, ref_cds_start=10, ref_aa="L", alt_aa="W"
        )

        assert result.impact != "UNKNOWN"
        assert result.impact in ("LOW", "MODERATE", "HIGH")

    def test_scientific_correctness_synonymous_has_low_impact(self):
        """Synonymous variants have LOW impact."""
        base_effect = create_test_effect(ref_aa="VLP", alt_aa="VLP")
        result = _classify_inframe_complex_effect(
            base_effect, ref_cds_start=5, ref_aa="VLP", alt_aa="VLP"
        )

        assert result.impact == "LOW"
        assert result.effect == "INFRAME_COMPLEX_SYNONYMOUS"

    def test_scientific_correctness_stop_codon_mutations_have_high_impact(self):
        """Stop codon mutations have HIGH impact."""
        base_effect1 = create_test_effect(ref_aa="*", alt_aa="E")
        result1 = _classify_inframe_complex_effect(
            base_effect1, ref_cds_start=100, ref_aa="*", alt_aa="E"
        )
        assert result1.impact == "HIGH"

        base_effect2 = create_test_effect(ref_aa="E", alt_aa="*")
        result2 = _classify_inframe_complex_effect(
            base_effect2, ref_cds_start=100, ref_aa="E", alt_aa="*"
        )
        assert result2.impact == "HIGH"
