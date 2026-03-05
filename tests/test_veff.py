"""Unit tests for variant effect annotation (veff.py)."""

import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from malariagen_data.veff import Annotator


def _make_annotator(seq: str) -> Annotator:
    """Build an Annotator over a single synthetic contig starting at position 1."""
    mock_genome = MagicMock()
    mock_genome["mock_chr"].__getitem__ = MagicMock(
        return_value=np.frombuffer(seq.encode(), dtype="S1")
    )

    end = len(seq)
    df_gff = pd.DataFrame(
        {
            "type": ["mRNA", "CDS"],
            "contig": ["mock_chr", "mock_chr"],
            "start": [1, 1],
            "end": [end, end],
            "strand": ["+", "+"],
            "ID": ["tx1", "cds1"],
            "Parent": [None, "tx1"],
            "phase": [0, 0],
        }
    )
    return Annotator(genome=mock_genome, genome_features=df_gff)


def get_effect(seq: str, pos: int, ref: str, alt: str) -> str:
    ann = _make_annotator(seq)
    variants = pd.DataFrame(
        {"position": [pos], "ref_allele": [ref], "alt_allele": [alt]}
    )
    return ann.get_effects(transcript="tx1", variants=variants)["effect"].iloc[0]


# Sequences used across tests:
#   ATG TGG CAG  →  M  W  Q   (no natural stop, for coding tests)
#   ATG TGG TGA  →  M  W  *   (stop at codon 3)
#   CTG TGG CAG  →  L  W  Q   (alternative start codon)
SEQ_MWQ = "ATGTGGCAG"
SEQ_MWstop = "ATGTGGTGA"
SEQ_LWQ = "CTGTGGCAG"  # CTG = alternative start, encodes Leu


class TestSnpEffects:
    def test_synonymous_coding(self):
        # pos 9: CAG → CAA, both = Gln (Q)
        assert get_effect(SEQ_MWQ, pos=9, ref="G", alt="A") == "SYNONYMOUS_CODING"

    def test_synonymous_stop(self):
        # pos 8: TGA → TAA, both are stop codons
        assert get_effect(SEQ_MWstop, pos=8, ref="G", alt="A") == "SYNONYMOUS_STOP"

    def test_synonymous_start(self):
        # pos 1: CTG → TTG, both encode Leu (L) at alternative start position
        assert get_effect(SEQ_LWQ, pos=1, ref="C", alt="T") == "SYNONYMOUS_START"

    def test_non_synonymous_coding(self):
        # pos 4: TGG → CGG, Trp → Arg — standard missense
        assert get_effect(SEQ_MWQ, pos=4, ref="T", alt="C") == "NON_SYNONYMOUS_CODING"

    def test_non_synonymous_start(self):
        # pos 1: CTG → ATG, Leu → Met at start codon position
        assert get_effect(SEQ_LWQ, pos=1, ref="C", alt="A") == "NON_SYNONYMOUS_START"

    def test_start_lost(self):
        # pos 1: ATG → TTG, Met → Leu — canonical start codon lost
        assert get_effect(SEQ_MWQ, pos=1, ref="A", alt="T") == "START_LOST"

    def test_stop_gained(self):
        # pos 7: CAG → TAG, Gln → Stop
        assert get_effect(SEQ_MWQ, pos=7, ref="C", alt="T") == "STOP_GAINED"

    def test_stop_lost(self):
        # pos 8: TGA → TCA, Stop → Ser
        assert get_effect(SEQ_MWstop, pos=8, ref="G", alt="C") == "STOP_LOST"
