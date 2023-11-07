Ag3
===

This page provides a curated list of functions and properties available in the malariagen_data API relating to *Anopheles gambiae* data.

.. currentmodule:: malariagen_data.ag3.Ag3

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
    count_samples
    lookup_sample
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

AIM data access
---------------
.. autosummary::
    :toctree: generated/

    aim_ids
    aim_variants
    aim_calls
    plot_aim_heatmap

CNV data access
---------------
.. autosummary::
    :toctree: generated/

    coverage_calls_analysis_ids
    cnv_hmm
    cnv_coverage_calls
    cnv_discordant_read_calls
    plot_cnv_hmm_coverage
    plot_cnv_hmm_heatmap
