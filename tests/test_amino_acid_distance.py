"""
Unit tests for the amino_acid_distance module.

These tests verify:
1. Known exact values from published reference tables.
2. Symmetry: d(A, B) == d(B, A) since distance is not directional.
3. Edge cases: same amino acid (no change), stop codons, invalid inputs.
"""


from malariagen_data.amino_acid_distance import grantham_score, sneath_dist


# Grantham score tests


class TestGranthamScore:
    def test_known_value_s_r(self):
        """S->R = 110, from Grantham (1974) Table 2."""
        assert grantham_score("S", "R") == 110

    def test_known_value_l_i(self):
        """L->I = 5, one of the smallest non-zero distances."""
        assert grantham_score("L", "I") == 5

    def test_known_value_c_w(self):
        """C->W = 215, one of the largest distances in the table."""
        assert grantham_score("C", "W") == 215

    def test_symmetry(self):
        """Grantham distance is symmetric: d(A,B) == d(B,A)."""
        assert grantham_score("S", "R") == grantham_score("R", "S")
        assert grantham_score("L", "I") == grantham_score("I", "L")
        assert grantham_score("C", "W") == grantham_score("W", "C")

    def test_same_amino_acid_returns_none(self):
        """When ref and alt are the same (synonymous), no distance applies."""
        # M->M is the only self-entry in the table with a value (67 in the tsv),
        # all other same-to-same lookups should return None (not in dict).
        assert grantham_score("L", "L") is None
        assert grantham_score("A", "A") is None

    def test_stop_codon_returns_none(self):
        """Stop codons (*) are not amino acids and should return None."""
        assert grantham_score("*", "L") is None
        assert grantham_score("L", "*") is None

    def test_none_input_returns_none(self):
        """None inputs (non-coding mutations) should safely return None."""
        assert grantham_score(None, "L") is None
        assert grantham_score("L", None) is None


# Sneath index tests


class TestSneathDist:
    def test_known_value_i_l(self):
        """I->L = 5, from Sneath matrix (very similar amino acids)."""
        assert sneath_dist("I", "L") == 5

    def test_known_value_r_d(self):
        """R->D = 39, from Sneath matrix."""
        assert sneath_dist("R", "D") == 39

    def test_symmetry(self):
        """Sneath distance is symmetric: d(A,B) == d(B,A)."""
        assert sneath_dist("I", "L") == sneath_dist("L", "I")
        assert sneath_dist("R", "D") == sneath_dist("D", "R")
        assert sneath_dist("W", "F") == sneath_dist("F", "W")

    def test_same_amino_acid_returns_zero(self):
        """Sneath table explicitly includes self-comparisons as 0."""
        assert sneath_dist("I", "I") == 0
        assert sneath_dist("L", "L") == 0
        assert sneath_dist("R", "R") == 0

    def test_stop_codon_returns_none(self):
        """Stop codons (*) are not amino acids and should return None."""
        assert sneath_dist("*", "L") is None
        assert sneath_dist("L", "*") is None

    def test_none_input_returns_none(self):
        """None inputs (non-coding mutations) should safely return None."""
        assert sneath_dist(None, "L") is None
        assert sneath_dist("L", None) is None
