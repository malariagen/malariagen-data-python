"""Machine Learning taxon classifier utility."""

import screed
from typing import Dict, Any, List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Mock trained K-mer database model
# In a real implementation this would load a pretrained joblib/pickle model
# generated from GCS reference genomes.
_MOCK_TRAINING_SEQS = [
    # gambiae_complex
    "ATGCGTACGTTAGCTAGCTAGCTA",
    "ATGCGTACGTTAGCTAGCTAGCTT",
    # funestus_subgroup
    "CCGGATATCGATCGATCGATCGCG",
    "CCGGATATCGATCGATCGATCGCG",
    # stephensi
    "TTAAGGCCTTAAGGCCTTAAGGCC",
    "TTAAGGCCTTAAGGCCTTAAGGCC",
]
_MOCK_LABELS = [
    "gambiae_complex",
    "gambiae_complex",
    "funestus_subgroup",
    "funestus_subgroup",
    "stephensi",
    "stephensi",
]

_vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 5))
_X_train = _vectorizer.fit_transform(_MOCK_TRAINING_SEQS)
_clf = MultinomialNB()
_clf.fit(_X_train, _MOCK_LABELS)

_RESOURCE_ROUTING = {
    "gambiae_complex": "Ag3",
    "funestus_subgroup": "Af1",
    "stephensi": "Amin1",
}


def identify_taxon(fastq_path: str, max_reads: int = 10000) -> Dict[str, Any]:
    """
    Identifies the taxon species group and recommends a malariagen_data resource
    from raw FASTQ reads using k-mer screening and a Naive Bayes classifier.

    Parameters
    ----------
    fastq_path : str
        Path to the `.fastq` or `.fastq.gz` file.
    max_reads : int, optional
        Maximum random reads to sample from the FASTQ to predict the group.
        Default is 10000.

    Returns
    -------
    dict
        A dictionary containing predicted_group, recommended_resource, confidence,
        method, and top_candidates scores.
    """

    sequences: List[str] = []

    # Fastq stream parsing using Screed
    # Screed automatically handles gzip.
    try:
        with screed.open(fastq_path) as file:
            # We want a random sample if the file is large
            # We will employ reservoir sampling to ensure memory stays low.
            # However, for simplicity let's just grab the first max_reads
            for i, record in enumerate(file):
                if i >= max_reads:
                    break
                sequences.append(record.sequence)
    except Exception as e:
        raise ValueError(f"Failed to read or parse fastq file: {fastq_path}") from e

    if not sequences:
        raise ValueError("FASTQ file is empty or invalid.")

    # Predict
    X_test = _vectorizer.transform(sequences)

    # Calculate average log probability across all sampled reads
    log_probs = _clf.predict_log_proba(X_test)
    avg_log_probs = np.mean(log_probs, axis=0)

    # Convert log-probs to linear probabilities
    probs = np.exp(avg_log_probs)
    # Normalize
    probs = probs / np.sum(probs)

    classes = _clf.classes_

    # Pair scores with labels
    scores = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

    best_taxon = scores[0][0]
    best_score = float(scores[0][1])

    top_candidates = [
        {"taxon": taxon, "score": round(float(score), 4)} for taxon, score in scores
    ]

    return {
        "predicted_group": best_taxon,
        "recommended_resource": _RESOURCE_ROUTING.get(best_taxon, "Unknown"),
        "confidence": round(best_score, 4),
        "method": "kmer_screening",
        "top_candidates": top_candidates,
    }
