import pytest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_veff_path = Path(__file__).resolve().parents[1] / "malariagen_data" / "veff.py"
_veff_spec = spec_from_file_location("veff_under_test", _veff_path)
assert _veff_spec is not None and _veff_spec.loader is not None
veff = module_from_spec(_veff_spec)
_veff_spec.loader.exec_module(veff)


@pytest.mark.parametrize(
    "ref_start,ref_stop,expected_effect,expected_impact",
    [
        (100, 101, "SPLICE_CORE", "HIGH"),
        (104, 105, "SPLICE_REGION", "MODERATE"),
        (120, 121, "INTRONIC", "MODIFIER"),
    ],
)
def test_intronic_indel_effects_are_classified(
    ref_start, ref_stop, expected_effect, expected_impact
):
    base_effect = veff.null_effect._replace(
        ref="AT",
        alt="A",
        ref_start=ref_start,
        ref_stop=ref_stop,
        strand="+",
    )

    effect = veff._get_within_intron_effect(base_effect=base_effect, intron=(100, 200))

    assert effect.effect == expected_effect
    assert effect.impact == expected_impact
    assert "TODO" not in effect.effect


def test_inframe_complex_cds_effect_is_classified(monkeypatch):
    # We only need to exercise branch logic here; codon details are not under test.
    monkeypatch.setattr(
        veff,
        "_get_aa_change",
        lambda ann, chrom, pos, ref, alt, cds, cdss: (
            0,
            1,
            0,
            "ATG",
            "ATG",
            1,
            "M",
            "M",
        ),
    )

    base_effect = veff.null_effect._replace(
        chrom="2L",
        pos=100,
        ref="AT",
        alt="ATGTA",
        ref_start=100,
        ref_stop=101,
        strand="+",
    )

    effect = veff._get_within_cds_effect(
        ann=None,
        base_effect=base_effect,
        cds=None,
        cdss=[],
    )

    assert effect.effect == "COMPLEX_CHANGE"
    assert effect.impact == "MODERATE"
    assert "TODO" not in effect.effect
