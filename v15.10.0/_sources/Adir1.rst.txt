Adir1
=====

This page provides a curated list of functions and properties available in the ``malariagen_data`` API
for data on mosquitoes from the *Anopheles dirus* complex.

To set up the API, use the following code::

    import malariagen_data
    adir1 = malariagen_data.Adir1()

All the functions below can then be accessed as methods on the ``adir1`` object. E.g., to call the
``sample_metadata()`` function, do::

    df_samples = adir1.sample_metadata()

For more information about the data and terms of use, please see the
`MalariaGEN website <https://www.malariagen.net/data>`_ or contact support@malariagen.net.

.. currentmodule:: malariagen_data.adir1.Adir1

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

Integrative genomics viewer (IGV)
---------------------------------
.. autosummary::
    :toctree: generated/

    igv
    view_alignments

SNP frequency analysis
----------------------
.. autosummary::
    :toctree: generated/

    snp_allele_frequencies
    snp_allele_frequencies_advanced
    aa_allele_frequencies
    aa_allele_frequencies_advanced
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
