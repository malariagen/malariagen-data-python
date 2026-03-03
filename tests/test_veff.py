import pandas as pd
import pytest

from malariagen_data.veff import Annotator


def test_annotator_get_children_returns_dataframe_for_single_child():
    genome_features = pd.DataFrame(
        [
            {
                "ID": "transcript_1",
                "Parent": "gene_1",
                "type": "mRNA",
                "contig": "2L",
                "start": 1,
                "end": 100,
                "strand": "+",
            },
            {
                "ID": "exon_1",
                "Parent": "transcript_1",
                "type": "exon",
                "contig": "2L",
                "start": 1,
                "end": 100,
                "strand": "+",
            },
        ]
    )
    ann = Annotator(genome={}, genome_features=genome_features)

    children = ann.get_children("transcript_1")

    assert isinstance(children, pd.DataFrame)
    assert children["ID"].to_list() == ["exon_1"]


def test_annotator_get_effects_raises_for_transcript_without_cds():
    genome_features = pd.DataFrame(
        [
            {
                "ID": "transcript_1",
                "Parent": "gene_1",
                "type": "mRNA",
                "contig": "2L",
                "start": 1,
                "end": 100,
                "strand": "+",
            },
            {
                "ID": "exon_1",
                "Parent": "transcript_1",
                "type": "exon",
                "contig": "2L",
                "start": 1,
                "end": 100,
                "strand": "+",
            },
        ]
    )
    ann = Annotator(genome={}, genome_features=genome_features)

    with pytest.raises(
        ValueError, match="has no CDS features; cannot compute coding variant effects"
    ):
        ann.get_effects("transcript_1", pd.DataFrame())
