Af1
===

This page provides a curated list of functions and properties available in the ``malariagen_data`` API
for data on mosquitoes from the *Anopheles funestus* subgroup.

To set up the API, use the following code::

    import malariagen_data
    af1 = malariagen_data.Af1()

All the functions below can then be accessed as methods on the ``af1`` object. E.g., to call the
``sample_metadata()`` function, do::

    df_samples = af1.sample_metadata()

For more information about the data and terms of use, please see the
`MalariaGEN Anopheles funestus genomic surveillance project <https://www.malariagen.net/projects/anopheles-funestus-genomic-surveillance-project>`_
home page.

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
    plot_sample_location_mapbox
    plot_sample_location_geo
    wgs_data_catalog
    cohorts

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
    biallelic_snp_calls
    biallelic_diplotypes
    biallelic_snps_to_plink

Haplotype data access
---------------------
.. autosummary::
    :toctree: generated/

    phasing_analysis_ids
    haplotypes
    haplotype_sites

CNV data access
---------------
.. autosummary::
    :toctree: generated/

    coverage_calls_analysis_ids
    cnv_hmm
    cnv_coverage_calls
    plot_cnv_hmm_coverage
    plot_cnv_hmm_heatmap
    gene_cnv

Note that CNV discordant read calls are not currently supported.

Integrative genomics viewer (IGV)
---------------------------------
.. autosummary::
    :toctree: generated/

    igv
    view_alignments

SNP and CNV frequency analysis
------------------------------
.. autosummary::
    :toctree: generated/

    snp_allele_frequencies
    snp_allele_frequencies_advanced
    aa_allele_frequencies
    aa_allele_frequencies_advanced
    gene_cnv_frequencies
    gene_cnv_frequencies_advanced
    haplotypes_frequencies
    haplotypes_frequencies_advanced
    plot_frequencies_heatmap
    plot_frequencies_time_series
    plot_frequencies_interactive_map

Principal components analysis (PCA)
-----------------------------------
.. autosummary::
    :toctree: generated/

    pca
    plot_pca_variance
    plot_pca_coords
    plot_pca_coords_3d

Genetic distance and neighbour-joining trees (NJT)
--------------------------------------------------
.. autosummary::
    :toctree: generated/

    plot_njt
    njt
    biallelic_diplotype_pairwise_distances

Heterozygosity analysis
-----------------------
.. autosummary::
    :toctree: generated/

    plot_heterozygosity
    roh_hmm
    plot_roh

Diversity analysis
------------------
.. autosummary::
    :toctree: generated/

    cohort_diversity_stats
    diversity_stats
    plot_diversity_stats

Genome-wide selection scans
---------------------------
.. autosummary::
    :toctree: generated/

    h12_calibration
    plot_h12_calibration
    h12_gwss
    plot_h12_gwss
    plot_h12_gwss_multi_panel
    plot_h12_gwss_multi_overlay
    h1x_gwss
    plot_h1x_gwss
    g123_calibration
    plot_g123_calibration
    g123_gwss
    plot_g123_gwss
    ihs_gwss
    plot_ihs_gwss
    xpehh_gwss
    plot_xpehh_gwss

Haplotype clustering and network analysis
-----------------------------------------
.. autosummary::
    :toctree: generated/

    plot_haplotype_clustering
    plot_haplotype_network
    haplotype_pairwise_distances

Diplotype clustering
--------------------
.. autosummary::
    :toctree: generated/

    plot_diplotype_clustering
    plot_diplotype_clustering_advanced

Fst analysis
------------
.. autosummary::
    :toctree: generated/

    average_fst
    pairwise_average_fst
    plot_pairwise_average_fst
    fst_gwss
    plot_fst_gwss
