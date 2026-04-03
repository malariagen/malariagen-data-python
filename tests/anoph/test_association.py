import pytest
import numpy as np
import xarray as xr
from malariagen_data.anoph.association import AnophelesAssociationAnalysis

class DummyAssociationAPI(AnophelesAssociationAnalysis):
    def __init__(self):
        # Skip cooperative multiple inheritance for pure logic testing
        pass


def test_variant_association_logic():
    """
    Test the statistical logic of variant_association by injecting a
    mock xarray Dataset with perfectly known phenotypic and genotypic correlations.
    """
    api = DummyAssociationAPI()
    
    # Let's mock the exact structure returned by phenotypes_with_snps()
    def mock_phenotypes_with_snps(
        region,
        sample_sets=None,
        sample_query=None,
        sample_query_options=None,
        cohort_size=None,
        min_cohort_size=None,
        max_cohort_size=None,
    ):
        # We will create 10 samples, 1 variant at position 1000
        # 5 samples ALIVE (1), 5 samples DEAD (0)
        # All ALIVE samples have alternate genotype (1/1) -> 2
        # All DEAD samples have reference genotype (0/0) -> 0
        samples = [f"SAM{i:03d}" for i in range(10)]
        phenotype_binary = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
        insecticide = np.array(["Permethrin"]*10)
        dose = np.array(["1x"]*10)
        phenotype = np.array(["alive"]*5 + ["dead"]*5)
        
        call_genotype = np.array([
            [
                # Sample 0..4 (Alive, has alt)
                [1, 1], [0, 1], [1, 0], [1, 1], [1, 1],
                # Sample 5..9 (Dead, ref only)
                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            ]
        ], dtype='i1')
        
        variant_position = np.array([1000])
        
        ds = xr.Dataset(
            {
                "phenotype_binary": (["samples"], phenotype_binary),
                "insecticide": (["samples"], insecticide),
                "dose": (["samples"], dose),
                "phenotype": (["samples"], phenotype),
                "call_genotype": (["variants", "samples", "ploidy"], call_genotype),
                "variant_position": (["variants"], variant_position)
            },
            coords={
                "samples": samples,
            }
        )
        return ds

    # Patch the method on the instance
    api.phenotypes_with_snps = mock_phenotypes_with_snps

    # Now call the mathematical feature
    res = api.variant_association(
        region="2L", 
        position=1000, 
        insecticide="Permethrin"
    )

    # In our exact scenario:
    # alt_alive = 5  (a)
    # ref_alive = 0  (b)
    # alt_dead = 0   (c)
    # ref_dead = 5   (d)
    
    assert res["region"] == "2L"
    assert res["position"] == 1000
    assert res["phenotype_positive_alt"] == 5
    assert res["phenotype_positive_ref"] == 0
    assert res["phenotype_negative_alt"] == 0
    assert res["phenotype_negative_ref"] == 5
    assert res["total_valid_samples"] == 10
    
    # With a perfect split of 5/5, the Odds Ratio is technically infinity,
    # and the P-value should be extremely significant (< 0.05)
    assert res["p_value"] < 0.05


def test_variant_association_not_found():
    api = DummyAssociationAPI()

    def mock_empty(*args, **kwargs):
        return xr.Dataset({"variant_position": (["variants"], [1000])}, coords={"samples": ["SAM001"]})
        
    api.phenotypes_with_snps = mock_empty
    
    with pytest.raises(ValueError, match="Variant position 9999 not found"):
        api.variant_association(region="2L", position=9999)
