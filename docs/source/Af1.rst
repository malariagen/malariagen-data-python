Af1
===

This page provides a curated list of functions and properties available in the malariagen_data API relating to *Anopheles funestus* data.

.. currentmodule:: malariagen_data.af1.Af1

Basic data access
-----------------
.. autosummary::
    :toctree: generated/

    releases
    sample_sets
    lookup_release
    lookup_study

Reference genome data access
----------------------------
.. autosummary::
    :toctree: generated/

    contigs
    genome_sequence
    genome_features
    plot_transcript
    plot_genes

Sample metadata access
----------------------
.. autosummary::
    :toctree: generated/

    sample_metadata
    add_extra_metadata
    clear_extra_metadata
    lookup_sample
    count_samples
    plot_samples_bar
    plot_samples_interactive_map
    wgs_data_catalog

SNP data access
---------------
.. autosummary::
    :toctree: generated/

    site_mask_ids
    snp_calls
    snp_allele_counts
    plot_snps
    site_annotations
    is_accessible

Haplotype data access
---------------------
.. autosummary::
    :toctree: generated/

    phasing_analysis_ids
    haplotypes

CNV data access
---------------
.. autosummary::
    :toctree: generated/

    coverage_calls_analysis_ids
    cnv_hmm
    cnv_coverage_calls
    plot_cnv_hmm_coverage
    plot_cnv_hmm_heatmap

Note that CNV discordant read calls are not currently supported.
