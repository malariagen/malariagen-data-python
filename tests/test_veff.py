"""Tests for variant effect functionality in veff.py"""

import pytest
from malariagen_data.veff import _get_within_cds_effect


class MockAnnotation:
    """Mock annotation object for testing"""
    def get_ref_allele_coords(self, chrom, pos, ref):
        return 0, len(ref)


def test_in_frame_complex_variation_with_codon_change():
    """Test complex in-frame variation with both MNP and INDEL causing codon changes"""
    # Create a mock base effect
    class MockEffect:
        def __init__(self):
            self.chrom = "2L"
            self.pos = 100
            self.ref = "ATGCGT"  # 6 nucleotides (2 codons)
            self.alt = "ATGAAAGT"  # 8 nucleotides (2 codons + 1 codon inserted)
            self.strand = "+"
            
        def _replace(self, **kwargs):
            new_effect = MockEffect()
            for key, value in kwargs.items():
                setattr(new_effect, key, value)
            return new_effect
    
    base_effect = MockEffect()
    ann = MockAnnotation()
    cds = [(0, 1000)]  # Simple CDS from 0 to 1000
    cdss = [(0, 1000)]
    
    # Mock the amino acid change function to return values indicating codon changes
    import malariagen_data.veff as veff_module
    original_get_aa_change = veff_module._get_aa_change
    
    def mock_get_aa_change(ann, chrom, pos, ref, alt, cds, cdss):
        # Return mock values indicating different amino acids
        return (
            0, 6, 0,  # ref_cds_start, ref_cds_stop, ref_start_phase
            "ATGCGT", "ATGAAAGT",  # ref_codon, alt_codon
            1,  # aa_pos
            "MS", "MKS"  # ref_aa, alt_aa (different - indicates codon change)
        )
    
    veff_module._get_aa_change = mock_get_aa_change
    
    try:
        effect = _get_within_cds_effect(ann, base_effect, cds, cdss)
        
        # Should be classified as CODON_CHANGE_PLUS_INDEL due to amino acid changes
        assert effect.effect == "CODON_CHANGE_PLUS_INDEL"
        assert effect.impact == "MODERATE"
    finally:
        # Restore original function
        veff_module._get_aa_change = original_get_aa_change


def test_in_frame_complex_variation_pure_insertion():
    """Test complex in-frame variation that is purely insertional without codon changes"""
    class MockEffect:
        def __init__(self):
            self.chrom = "2L"
            self.pos = 100
            self.ref = "ATGCGT"  # 6 nucleotides (2 codons)
            self.alt = "ATGCGTAAAGT"  # 11 nucleotides (2 codons + 1 codon inserted)
            self.strand = "+"
            
        def _replace(self, **kwargs):
            new_effect = MockEffect()
            for key, value in kwargs.items():
                setattr(new_effect, key, value)
            return new_effect
    
    base_effect = MockEffect()
    ann = MockAnnotation()
    cds = [(0, 1000)]
    cdss = [(0, 1000)]
    
    import malariagen_data.veff as veff_module
    original_get_aa_change = veff_module._get_aa_change
    
    def mock_get_aa_change(ann, chrom, pos, ref, alt, cds, cdss):
        # Return mock values indicating same amino acids with insertion
        return (
            0, 6, 0,  # ref_cds_start, ref_cds_stop, ref_start_phase
            "ATGCGT", "ATGCGTAAAGT",  # ref_codon, alt_codon
            1,  # aa_pos
            "MS", "MSK"  # ref_aa, alt_aa (original amino acids preserved + new one)
        )
    
    veff_module._get_aa_change = mock_get_aa_change
    
    try:
        effect = _get_within_cds_effect(ann, base_effect, cds, cdss)
        
        # Should be classified as CODON_INSERTION since alt > ref and no codon changes
        assert effect.effect == "CODON_INSERTION"
        assert effect.impact == "MODERATE"
    finally:
        veff_module._get_aa_change = original_get_aa_change


def test_in_frame_complex_variation_pure_deletion():
    """Test complex in-frame variation that is purely deletional without codon changes"""
    class MockEffect:
        def __init__(self):
            self.chrom = "2L"
            self.pos = 100
            self.ref = "ATGCGTAAAGT"  # 11 nucleotides (3 codons + partial)
            self.alt = "ATGCGT"  # 6 nucleotides (2 codons)
            self.strand = "+"
            
        def _replace(self, **kwargs):
            new_effect = MockEffect()
            for key, value in kwargs.items():
                setattr(new_effect, key, value)
            return new_effect
    
    base_effect = MockEffect()
    ann = MockAnnotation()
    cds = [(0, 1000)]
    cdss = [(0, 1000)]
    
    import malariagen_data.veff as veff_module
    original_get_aa_change = veff_module._get_aa_change
    
    def mock_get_aa_change(ann, chrom, pos, ref, alt, cds, cdss):
        # Return mock values indicating same amino acids with deletion
        return (
            0, 6, 0,  # ref_cds_start, ref_cds_stop, ref_start_phase
            "ATGCGTAAAGT", "ATGCGT",  # ref_codon, alt_codon
            1,  # aa_pos
            "MSK", "MS"  # ref_aa, alt_aa (original amino acids, one deleted)
        )
    
    veff_module._get_aa_change = mock_get_aa_change
    
    try:
        effect = _get_within_cds_effect(ann, base_effect, cds, cdss)
        
        # Should be classified as CODON_DELETION since ref > alt and no codon changes
        assert effect.effect == "CODON_DELETION"
        assert effect.impact == "MODERATE"
    finally:
        veff_module._get_aa_change = original_get_aa_change
