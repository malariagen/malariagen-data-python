Af1 API
=======

This page provides a curated list of functions and properties available in the malariagen_data API relating to *Anopheles funestus* data.

.. currentmodule:: malariagen_data.af1.Af1

Basic data access
-----------------
.. autosummary::
    :toctree: generated/

    client_location
    config
    lookup_release
    open_file
    read_files
    releases
    results_cache_get
    results_cache_set
    sample_sets

Sample metadata access
----------------------
.. autosummary::
    :toctree: generated/

    add_extra_metadata
    aim_metadata
    clear_extra_metadata
    cohorts_metadata
    count_samples
    general_metadata
    lookup_sample
    plot_samples_bar
    plot_samples_interactive_map
    sample_metadata
    wgs_data_catalog

SNP data access
---------------
.. autosummary::
    :toctree: generated/

    is_accessible
    open_site_annotations
    open_site_filters
    open_snp_genotypes
    open_snp_sites
    plot_snps
    plot_snps_track
    site_annotations
    site_filters
    site_mask_ids
    snp_allele_counts
    snp_calls
    snp_dataset
    snp_genotypes
    snp_sites
    snp_variants

Haplotype data access
---------------------
.. autosummary::
    :toctree: generated/

    haplotypes
    open_haplotypes
    open_haplotype_sites
    phasing_analysis_ids


CNV data access
---------------
.. autosummary::
    :toctree: generated/

    coverage_calls_analysis_ids
    cnv_coverage_calls
    cnv_discordant_read_calls
    cnv_hmm
    open_cnv_coverage_calls
    open_cnv_discordant_read_calls
    open_cnv_hmm
    plot_cnv_hmm_coverage
    plot_cnv_hmm_coverage_track
    plot_cnv_hmm_heatmap
    plot_cnv_hmm_heatmap_track
