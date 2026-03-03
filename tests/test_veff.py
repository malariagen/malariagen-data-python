import pandas as pd

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
