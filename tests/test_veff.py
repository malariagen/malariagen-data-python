"""Unit tests for malariagen_data.veff — Annotator and helper functions.

Fixtures use in-memory zarr stores and pandas DataFrames; no GCS access required.
Codon table: standard NCBI Table 1 (default for Bio.Seq.translate()).

Forward-strand genome (chr1):
  pos 1-3:   AAA  (flanking)
  pos 4-6:   ATG  = Met (M)
  pos 7-9:   GCC  = Ala (A)
  pos 10-12: TTA  = Leu (L)
  pos 13-15: CAG  = Gln (Q)
  pos 16-18: TGA  = Stop (*)
  pos 19-38: intron (20 bp)
  pos 39-41: CAG  = Gln (Q)  [exon 2]
  pos 42-44: TGA  = Stop (*) [exon 2]
  pos 45-54: AAA... (flanking / 3' UTR)

Reverse-strand genome (chr2), gene at pos 4-15 (-):
  Forward seq: TCATGCATGCAT
  RevComp    : ATGCATGCATGA = ATG|CAT|GCA|TGA = M|H|A|*
"""

import numpy as np
import pandas as pd
import pytest
import zarr

from malariagen_data.veff import Annotator

# ── Genome sequences ──────────────────────────────────────────────────────────

FWD_SEQ = (
    "AAA"
    "ATG"
    "GCC"
    "TTA"
    "CAG"
    "TGA"  # M A L Q *  (pos 4-18)
    "AAAAAAAAAAAAAAAAAAAA"  # intron      (pos 19-38)
    "CAG"
    "TGA"  # Q *         (pos 39-44, exon 2)
    "AAAAAAAAAA"  # 3' UTR      (pos 45-54)
)

REV_SEQ = "AAA" "TCATGCATGCAT" "AAAA"  # minus-strand gene at pos 4-15


# ── Fixture helpers ───────────────────────────────────────────────────────────


def make_genome(*chrom_seq_pairs):
    """Return an in-memory zarr store with one |S1 array per chromosome."""
    store = zarr.group()
    for chrom, seq in chrom_seq_pairs:
        store.create_dataset(
            chrom, data=np.frombuffer(seq.encode("ascii"), dtype="|S1")
        )
    return store


def make_features(rows):
    """Build a GFF3-style features DataFrame.

    Required columns: contig (str), type (str), start (int64), end (int64),
    strand (str), ID (str), Parent (str).  All rows must have end > start.
    """
    df = pd.DataFrame(rows)
    df[["start", "end"]] = df[["start", "end"]].astype(np.int64)
    return df


def make_variants(rows):
    """Build a variants DataFrame with columns: position (int64), ref_allele, alt_allele."""
    df = pd.DataFrame(rows)
    df["position"] = df["position"].astype(np.int64)
    return df


# ── Shared feature sets ───────────────────────────────────────────────────────

# Single-exon forward-strand gene, CDS pos 4-18
FEATURES_BASIC_FWD = make_features(
    [
        {
            "contig": "chr1",
            "type": "mRNA",
            "start": 4,
            "end": 18,
            "strand": "+",
            "ID": "tx1",
            "Parent": "gene1",
        },
        {
            "contig": "chr1",
            "type": "exon",
            "start": 4,
            "end": 18,
            "strand": "+",
            "ID": "exon1",
            "Parent": "tx1",
        },
        {
            "contig": "chr1",
            "type": "CDS",
            "start": 4,
            "end": 18,
            "strand": "+",
            "ID": "cds1",
            "Parent": "tx1",
        },
    ]
)

# Gene with 5' UTR (pos 1-3), CDS (pos 4-18), 3' UTR (pos 45-54)
FEATURES_UTR_FWD = make_features(
    [
        {
            "contig": "chr1",
            "type": "mRNA",
            "start": 1,
            "end": 54,
            "strand": "+",
            "ID": "tx_utr",
            "Parent": "gene_utr",
        },
        {
            "contig": "chr1",
            "type": "exon",
            "start": 1,
            "end": 54,
            "strand": "+",
            "ID": "exon_u",
            "Parent": "tx_utr",
        },
        {
            "contig": "chr1",
            "type": "five_prime_UTR",
            "start": 1,
            "end": 3,
            "strand": "+",
            "ID": "utr5",
            "Parent": "tx_utr",
        },
        {
            "contig": "chr1",
            "type": "CDS",
            "start": 4,
            "end": 18,
            "strand": "+",
            "ID": "cds_u",
            "Parent": "tx_utr",
        },
        {
            "contig": "chr1",
            "type": "three_prime_UTR",
            "start": 45,
            "end": 54,
            "strand": "+",
            "ID": "utr3",
            "Parent": "tx_utr",
        },
    ]
)

# Two-exon gene: exon1=4-18, intron=19-38, exon2=39-44
FEATURES_SPLICE_FWD = make_features(
    [
        {
            "contig": "chr1",
            "type": "mRNA",
            "start": 4,
            "end": 44,
            "strand": "+",
            "ID": "tx_sp",
            "Parent": "gene_sp",
        },
        {
            "contig": "chr1",
            "type": "exon",
            "start": 4,
            "end": 18,
            "strand": "+",
            "ID": "exon_s1",
            "Parent": "tx_sp",
        },
        {
            "contig": "chr1",
            "type": "CDS",
            "start": 4,
            "end": 18,
            "strand": "+",
            "ID": "cds_s1",
            "Parent": "tx_sp",
        },
        {
            "contig": "chr1",
            "type": "exon",
            "start": 39,
            "end": 44,
            "strand": "+",
            "ID": "exon_s2",
            "Parent": "tx_sp",
        },
        {
            "contig": "chr1",
            "type": "CDS",
            "start": 39,
            "end": 44,
            "strand": "+",
            "ID": "cds_s2",
            "Parent": "tx_sp",
        },
    ]
)

# Single-exon minus-strand gene at chr2:4-15
FEATURES_REV = make_features(
    [
        {
            "contig": "chr2",
            "type": "mRNA",
            "start": 4,
            "end": 15,
            "strand": "-",
            "ID": "tx_rev",
            "Parent": "gene_rev",
        },
        {
            "contig": "chr2",
            "type": "exon",
            "start": 4,
            "end": 15,
            "strand": "-",
            "ID": "exon_rev",
            "Parent": "tx_rev",
        },
        {
            "contig": "chr2",
            "type": "CDS",
            "start": 4,
            "end": 15,
            "strand": "-",
            "ID": "cds_rev",
            "Parent": "tx_rev",
        },
    ]
)


def _run(ann, transcript, position, ref_allele, alt_allele):
    variants = make_variants(
        [{"position": position, "ref_allele": ref_allele, "alt_allele": alt_allele}]
    )
    result = ann.get_effects(transcript, variants)
    assert len(result) == 1
    return result.iloc[0]


# ── TestAnnotatorHelpers ──────────────────────────────────────────────────────


class TestAnnotatorHelpers:
    def setup_method(self):
        self.ann = Annotator(make_genome(("chr1", FWD_SEQ)), FEATURES_BASIC_FWD.copy())

    def test_get_ref_seq_correct(self):
        assert self.ann.get_ref_seq("chr1", 4, 6) == "ATG"

    def test_get_ref_seq_single_base(self):
        assert self.ann.get_ref_seq("chr1", 7, 7) == "G"

    def test_get_ref_seq_caches_contig(self):
        self.ann.get_ref_seq("chr1", 4, 6)
        assert "chr1" in self.ann._genome_cache
        # Deleting the zarr array confirms the cache is served on the second call.
        del self.ann._genome["chr1"]
        assert self.ann.get_ref_seq("chr1", 4, 6) == "ATG"

    def test_get_ref_allele_coords_correct(self):
        assert self.ann.get_ref_allele_coords("chr1", 4, "ATG") == (4, 6)

    def test_get_ref_allele_coords_mismatch_raises(self):
        # Genome at pos 4-6 is "ATG"; passing "CCC" must raise.
        with pytest.raises(ValueError, match="Reference allele does not match"):
            self.ann.get_ref_allele_coords("chr1", 4, "CCC")

    def test_get_feature_by_id(self):
        feat = self.ann.get_feature("tx1")
        assert feat["type"] == "mRNA"

    def test_get_children_returns_dataframe(self):
        # Even a transcript with a single child must return a DataFrame, not a Series.
        children = self.ann.get_children("tx1")
        assert isinstance(children, pd.DataFrame)
        assert len(children) == 2


# ── TestAnnotatorGetEffects_ForwardStrand ─────────────────────────────────────
#
# Gene tx1: chr1:4-18 (+)  —  codons ATG|GCC|TTA|CAG|TGA = M|A|L|Q|*


class TestAnnotatorGetEffects_ForwardStrand:
    def setup_method(self):
        self.ann = Annotator(make_genome(("chr1", FWD_SEQ)), FEATURES_BASIC_FWD.copy())

    def test_synonymous_coding(self):
        # TTA→TTG at pos 12 (3rd base): Leu→Leu
        row = _run(self.ann, "tx1", 12, "A", "G")
        assert row["effect"] == "SYNONYMOUS_CODING"
        assert row["impact"] == "LOW"
        assert row["ref_aa"] == row["alt_aa"] == "L"
        assert row["grantham_score"] is None
        # Sneath stores self-comparisons as 0 (L→L = 0), not None.
        assert row["sneath_score"] == 0

    def test_non_synonymous_coding(self):
        # TTA→GTA at pos 10: Leu→Val, Grantham=32, Sneath=7
        row = _run(self.ann, "tx1", 10, "T", "G")
        assert row["effect"] == "NON_SYNONYMOUS_CODING"
        assert row["impact"] == "MODERATE"
        assert row["ref_aa"] == "L"
        assert row["alt_aa"] == "V"
        assert row["grantham_score"] == 32
        assert row["sneath_score"] == 7

    def test_stop_gained(self):
        # CAG→TAG at pos 13: Gln→*
        row = _run(self.ann, "tx1", 13, "C", "T")
        assert row["effect"] == "STOP_GAINED"
        assert row["impact"] == "HIGH"
        assert row["ref_aa"] == "Q"
        assert row["alt_aa"] == "*"
        assert row["grantham_score"] is None

    def test_stop_lost(self):
        # TGA→CGA at pos 16: *→Arg
        row = _run(self.ann, "tx1", 16, "T", "C")
        assert row["effect"] == "STOP_LOST"
        assert row["impact"] == "HIGH"
        assert row["ref_aa"] == "*"

    def test_start_lost(self):
        # ATG→GTG at pos 4: Met→Val at start codon
        row = _run(self.ann, "tx1", 4, "A", "G")
        assert row["effect"] == "START_LOST"
        assert row["impact"] == "HIGH"
        assert row["ref_aa"] == "M"

    def test_synonymous_stop(self):
        # TGA→TAA at pos 17: *→* (both Stop)
        row = _run(self.ann, "tx1", 17, "G", "A")
        assert row["effect"] == "SYNONYMOUS_STOP"
        assert row["impact"] == "LOW"
        assert row["ref_aa"] == row["alt_aa"] == "*"

    def test_frame_shift_insertion(self):
        # Insert 1 bp inside CDS: net change = +1, 1 % 3 ≠ 0
        row = _run(self.ann, "tx1", 7, "G", "GA")
        assert row["effect"] == "FRAME_SHIFT"
        assert row["impact"] == "HIGH"

    def test_frame_shift_deletion(self):
        # Delete 1 bp inside CDS: net change = -1, 1 % 3 ≠ 0
        row = _run(self.ann, "tx1", 7, "GC", "G")
        assert row["effect"] == "FRAME_SHIFT"
        assert row["impact"] == "HIGH"

    def test_codon_insertion_at_boundary(self):
        # Insert 3 bp at pos 7 (start of GCC codon, phase=0).
        # alt_codon first triplet = GCA (Ala) = ref_aa[0] → anchor unchanged.
        row = _run(self.ann, "tx1", 7, "G", "GCAT")
        assert row["effect"] == "CODON_INSERTION"
        assert row["impact"] == "MODERATE"

    def test_codon_change_plus_insertion(self):
        # Insert 3 bp at pos 11 (mid-codon): anchor codon changes Leu→Phe.
        row = _run(self.ann, "tx1", 11, "T", "TCAT")
        assert row["effect"] == "CODON_CHANGE_PLUS_CODON_INSERTION"
        assert row["impact"] == "MODERATE"

    def test_codon_change_plus_deletion(self):
        # Delete 3 bp (len(ref)=4, len(alt)=1, net=-3) at pos 7, anchor changes.
        row = _run(self.ann, "tx1", 7, "GCCT", "G")
        assert row["effect"] == "CODON_CHANGE_PLUS_CODON_DELETION"
        assert row["impact"] == "MODERATE"

    def test_codon_deletion_variant(self):
        # Delete 3 bp at pos 10 (net=-3), anchor codon changes Leu→Cys.
        row = _run(self.ann, "tx1", 10, "TTAC", "T")
        assert row["effect"] == "CODON_CHANGE_PLUS_CODON_DELETION"
        assert row["impact"] == "MODERATE"

    def test_codon_change_mnp(self):
        # MNP: same-length substitution of whole codon TTA→GGG
        row = _run(self.ann, "tx1", 10, "TTA", "GGG")
        assert row["effect"] == "CODON_CHANGE"
        assert row["impact"] == "MODERATE"


# ── TestAnnotatorGetEffects_ReverseStrand ─────────────────────────────────────
#
# Gene tx_rev: chr2:4-15 (-)  —  codons M|H|A|* (RevComp of forward seq)
#
# Codon-to-forward-position mapping (minus strand reads right→left):
#   Codon 1 (ATG=M): fwd pos 13-15  |  Codon 2 (CAT=H): fwd pos 10-12
#   Codon 3 (GCA=A): fwd pos 7-9   |  Codon 4 (TGA=*): fwd pos 4-6
#
# Synonymous change (H→H, CAT→CAC):
#   CAT[2]=T ← complement of fwd pos 10 ('A').  CAC needs complement(C)=G → fwd pos 10 A→G.
#
# Non-synonymous change (A→V, GCA→GTA):
#   GCA[1]=C ← complement of fwd pos 8 ('G').  GTA needs complement(T)=A → fwd pos 8 G→A.


class TestAnnotatorGetEffects_ReverseStrand:
    def setup_method(self):
        self.ann = Annotator(make_genome(("chr2", REV_SEQ)), FEATURES_REV.copy())

    def test_rev_synonymous_coding(self):
        # fwd pos 10 A→G: CAT→CAC (His→His) on minus strand
        row = _run(self.ann, "tx_rev", 10, "A", "G")
        assert row["effect"] == "SYNONYMOUS_CODING"
        assert row["ref_aa"] == row["alt_aa"] == "H"

    def test_rev_non_synonymous_coding(self):
        # fwd pos 8 G→A: GCA→GTA (Ala→Val) on minus strand
        row = _run(self.ann, "tx_rev", 8, "G", "A")
        assert row["effect"] == "NON_SYNONYMOUS_CODING"
        assert row["ref_aa"] == "A"
        assert row["alt_aa"] == "V"


# ── TestAnnotatorGetEffects_UTR ───────────────────────────────────────────────


class TestAnnotatorGetEffects_UTR:
    def setup_method(self):
        self.ann = Annotator(make_genome(("chr1", FWD_SEQ)), FEATURES_UTR_FWD.copy())

    def test_five_prime_utr(self):
        row = _run(self.ann, "tx_utr", 2, "A", "T")
        assert row["effect"] == "FIVE_PRIME_UTR"
        assert row["impact"] == "LOW"

    def test_three_prime_utr(self):
        row = _run(self.ann, "tx_utr", 50, "A", "T")
        assert row["effect"] == "THREE_PRIME_UTR"
        assert row["impact"] == "LOW"


# ── TestAnnotatorGetEffects_Intron ────────────────────────────────────────────
#
# Intron: pos 19-38 (20 bp).  Splice distances for a + strand SNP at pos p:
#   5' dist = p - 18  |  3' dist = 39 - p
#   min_dist = min(5' dist, 3' dist)
#
#   pos 19 → min=1  → SPLICE_CORE
#   pos 20 → min=2  → SPLICE_CORE
#   pos 38 → min=1  → SPLICE_CORE  (3' end)
#   pos 23 → min=5  → SPLICE_REGION
#   pos 27 → min=9  → INTRONIC


class TestAnnotatorGetEffects_Intron:
    def setup_method(self):
        self.ann = Annotator(make_genome(("chr1", FWD_SEQ)), FEATURES_SPLICE_FWD.copy())

    def test_splice_core_5prime(self):
        row = _run(self.ann, "tx_sp", 19, "A", "T")
        assert row["effect"] == "SPLICE_CORE"
        assert row["impact"] == "HIGH"

    def test_splice_core_5prime_pos2(self):
        row = _run(self.ann, "tx_sp", 20, "A", "T")
        assert row["effect"] == "SPLICE_CORE"

    def test_splice_core_3prime(self):
        row = _run(self.ann, "tx_sp", 38, "A", "T")
        assert row["effect"] == "SPLICE_CORE"

    def test_splice_region(self):
        row = _run(self.ann, "tx_sp", 23, "A", "T")
        assert row["effect"] == "SPLICE_REGION"
        assert row["impact"] == "MODERATE"

    def test_intronic(self):
        row = _run(self.ann, "tx_sp", 27, "A", "T")
        assert row["effect"] == "INTRONIC"
        assert row["impact"] == "MODIFIER"


# ── TestAnnotatorGetEffects_ErrorCases ───────────────────────────────────────


class TestAnnotatorGetEffects_ErrorCases:
    def setup_method(self):
        self.ann = Annotator(make_genome(("chr1", FWD_SEQ)), FEATURES_BASIC_FWD.copy())

    def test_variant_outside_transcript_raises(self):
        # tx1 spans pos 4-18; pos 1 is outside.
        with pytest.raises(ValueError, match="fall outside transcript boundaries"):
            _run(self.ann, "tx1", 1, "A", "T")

    def test_no_cds_or_utr_raises(self):
        # A transcript with only an exon (no CDS, no UTR) is non-coding.
        features = make_features(
            [
                {
                    "contig": "chr1",
                    "type": "mRNA",
                    "start": 4,
                    "end": 18,
                    "strand": "+",
                    "ID": "tx_nc",
                    "Parent": "gene_nc",
                },
                {
                    "contig": "chr1",
                    "type": "exon",
                    "start": 4,
                    "end": 18,
                    "strand": "+",
                    "ID": "exon_nc",
                    "Parent": "tx_nc",
                },
            ]
        )
        ann = Annotator(make_genome(("chr1", FWD_SEQ)), features)
        with pytest.raises(ValueError, match="has no CDS or UTR children"):
            _run(ann, "tx_nc", 10, "T", "G")


# ── TestGranthamSneathScores ──────────────────────────────────────────────────


class TestGranthamSneathScores:
    def setup_method(self):
        self.ann = Annotator(make_genome(("chr1", FWD_SEQ)), FEATURES_BASIC_FWD.copy())

    def test_non_synonymous_has_scores(self):
        row = _run(self.ann, "tx1", 10, "T", "G")  # L→V
        assert row["grantham_score"] > 0
        assert row["sneath_score"] > 0

    def test_synonymous_grantham_none(self):
        # L→L: Grantham has no self-entry for Leu.
        row = _run(self.ann, "tx1", 12, "A", "G")
        assert row["grantham_score"] is None

    def test_stop_gained_no_scores(self):
        row = _run(self.ann, "tx1", 13, "C", "T")  # Q→*
        assert row["grantham_score"] is None
        assert row["sneath_score"] is None

    def test_stop_lost_no_scores(self):
        row = _run(self.ann, "tx1", 16, "T", "C")  # *→R
        assert row["grantham_score"] is None
        assert row["sneath_score"] is None


# ── TestSynonymousStart ───────────────────────────────────────────────────────


class TestSynonymousStart:
    """
    The SYNONYMOUS_START branch in `_get_within_cds_effect` requires
    ref_cds_start == 0 and ref_aa == alt_aa. No SNP on ATG produces a
    synonymous Met under the standard NCBI codon table, so this branch
    is unreachable in practice. A test should be added here if
    non-standard codon tables are ever supported.
    """

    def test_synonymous_start_is_unreachable_with_standard_codon_table(self):
        pass


# ── TestGenomeCacheDefaultMaxsize ────────────────────────────────────────────


class TestGenomeCacheDefaultMaxsize:
    """Verify that the default cache maxsize is 5."""

    def test_default_maxsize(self):
        genome = make_genome(("chr1", FWD_SEQ))
        ann = Annotator(genome=genome, genome_features=FEATURES_BASIC_FWD.copy())
        cache_info = ann._load_genome_seq.cache_info()
        assert cache_info.maxsize == 5


# ── TestGenomeCacheLRUEviction ───────────────────────────────────────────────


class TestGenomeCacheLRUEviction:
    """Verify that the LRU cache evicts the oldest entry when full."""

    def test_eviction(self):
        contigs = [("chr1", FWD_SEQ), ("chr2", REV_SEQ), ("chr3", FWD_SEQ)]
        genome = make_genome(*contigs)
        ann = Annotator(
            genome=genome,
            genome_features=FEATURES_BASIC_FWD.copy(),
            genome_cache_maxsize=2,
        )

        # Load all three contigs in order.
        for chrom, _ in contigs:
            ann._load_genome_seq(chrom)

        info = ann._load_genome_seq.cache_info()
        # Only 2 entries should remain (chr2 and chr3).
        assert info.currsize == 2
        # 3 total misses (each first access is a miss).
        assert info.misses == 3

        # Accessing chr1 again should be a miss because it was evicted.
        ann._load_genome_seq("chr1")
        info = ann._load_genome_seq.cache_info()
        assert info.misses == 4


# ── TestClearGenomeCache ─────────────────────────────────────────────────────


class TestClearGenomeCache:
    """Verify that clear_genome_cache() empties the cache."""

    def test_clear(self):
        genome = make_genome(("chr1", FWD_SEQ), ("chr2", REV_SEQ))
        ann = Annotator(genome=genome, genome_features=FEATURES_BASIC_FWD.copy())

        ann._load_genome_seq("chr1")
        ann._load_genome_seq("chr2")
        assert ann._load_genome_seq.cache_info().currsize == 2

        ann.clear_genome_cache()
        assert ann._load_genome_seq.cache_info().currsize == 0


# ── TestGenomeCacheUnbounded ─────────────────────────────────────────────────


class TestGenomeCacheUnbounded:
    """Verify that maxsize=None gives an unbounded cache."""

    def test_unbounded(self):
        contigs = [(f"chr{i}", FWD_SEQ) for i in range(20)]
        genome = make_genome(*contigs)
        ann = Annotator(
            genome=genome,
            genome_features=FEATURES_BASIC_FWD.copy(),
            genome_cache_maxsize=None,
        )

        for chrom, _ in contigs:
            ann._load_genome_seq(chrom)

        info = ann._load_genome_seq.cache_info()
        assert info.maxsize is None
        assert info.currsize == 20


# ── TestPerInstanceCacheIsolation ────────────────────────────────────────────


class TestPerInstanceCacheIsolation:
    """Verify that two Annotator instances have independent caches."""

    def test_isolation(self):
        genome = make_genome(("chr1", FWD_SEQ), ("chr2", REV_SEQ))
        features = FEATURES_BASIC_FWD.copy()

        ann1 = Annotator(genome=genome, genome_features=features)
        ann2 = Annotator(genome=genome, genome_features=features)

        ann1._load_genome_seq("chr1")
        assert ann1._load_genome_seq.cache_info().currsize == 1
        assert ann2._load_genome_seq.cache_info().currsize == 0

        ann2._load_genome_seq("chr2")
        assert ann1._load_genome_seq.cache_info().currsize == 1
        assert ann2._load_genome_seq.cache_info().currsize == 1

        ann1.clear_genome_cache()
        assert ann1._load_genome_seq.cache_info().currsize == 0
        assert ann2._load_genome_seq.cache_info().currsize == 1
