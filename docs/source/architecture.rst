Package architecture
=====================

Each dataset class in ``malariagen_data`` (``Ag3``, ``Af1``, ``As1``,
``Amin1``, ``Adir1``, ``Adar1``) is a thin, dataset-specific subclass of
:class:`malariagen_data.AnophelesDataResource`, which itself is composed
from a number of mixin classes, each providing a related group of
functionality (e.g. SNP data access, PCA, selection scans, CNV data). All
six dataset classes share the exact same mixin structure - only the
top-level dataset class and its configuration (which data are available,
default analyses, and so on) differ between them.

The mixin structure is split below into
several smaller ones, each showing one area of functionality. A few classes
- particularly :class:`~malariagen_data.anoph.snp_data.AnophelesSnpData` and
:class:`~malariagen_data.anoph.hap_data.AnophelesHapData` - appear in more
than one diagram, as multiple types of functionality
build on them. Thus,  each diagram is
self-contained and doesn't require cross-referencing another one to make
sense.

Every box links through to its own reference page (its own existing page,
for the six dataset classes - see :doc:`Ag3`, :doc:`Af1`, :doc:`As1`,
:doc:`Amin1`, :doc:`Adir1`, :doc:`Adar1` - or a generated page listing just
what it contributes directly, for the mixins), which in turn links on to
its source code.

See :doc:`full_inheritance` for the same structure as a single, complete
diagram rather than split by topic.

Composition overview
---------------------
How the six dataset classes and :class:`~malariagen_data.AnophelesDataResource`
relate to the top-level mixins - each of which is expanded further in one
of the diagrams below.

.. inheritance-diagram:: malariagen_data.ag3.Ag3 malariagen_data.af1.Af1 malariagen_data.as1.As1 malariagen_data.amin1.Amin1 malariagen_data.adir1.Adir1 malariagen_data.adar1.Adar1
    :parts: 1
    :top-classes: malariagen_data.anoph.dipclust.AnophelesDipClustAnalysis, malariagen_data.anoph.hapclust.AnophelesHapClustAnalysis, malariagen_data.anoph.xpehh.AnophelesXpehhAnalysis, malariagen_data.anoph.h1x.AnophelesH1XAnalysis, malariagen_data.anoph.h12.AnophelesH12Analysis, malariagen_data.anoph.g123.AnophelesG123Analysis, malariagen_data.anoph.fst.AnophelesFstAnalysis, malariagen_data.anoph.heterozygosity.AnophelesHetAnalysis, malariagen_data.anoph.hap_frq.AnophelesHapFrequencyAnalysis, malariagen_data.anoph.distance.AnophelesDistanceAnalysis, malariagen_data.anoph.pca.AnophelesPca, malariagen_data.anoph.to_plink.PlinkConverter, malariagen_data.anoph.ld.AnophelesLdAnalysis, malariagen_data.anoph.to_vcf.SnpVcfExporter, malariagen_data.anoph.igv.AnophelesIgv, malariagen_data.anoph.karyotype.AnophelesKaryotypeAnalysis, malariagen_data.anoph.aim_data.AnophelesAimData, malariagen_data.anoph.hap_data.AnophelesHapData, malariagen_data.anoph.snp_data.AnophelesSnpData, malariagen_data.anoph.sample_metadata.AnophelesSampleMetadata, malariagen_data.anoph.genome_features.AnophelesGenomeFeaturesData, malariagen_data.anoph.genome_sequence.AnophelesGenomeSequenceData, malariagen_data.anoph.describe.AnophelesDescribe, malariagen_data.anoph.base.AnophelesBase, malariagen_data.anoph.phenotypes.AnophelesPhenotypeData

Core data access
----------------
The foundational classes that everything else in the other diagrams
ultimately builds on: SNP, haplotype, AIM and CNV data access, sample
metadata, and the reference genome.

.. inheritance-diagram:: malariagen_data.anoph.snp_data.AnophelesSnpData malariagen_data.anoph.hap_data.AnophelesHapData malariagen_data.anoph.aim_data.AnophelesAimData malariagen_data.anoph.cnv_data.AnophelesCnvData
    :parts: 1
    :top-classes: object

SNP-based analyses
-------------------
Everything built directly on :class:`~malariagen_data.anoph.snp_data.AnophelesSnpData`
(PCA, Fst, LD, heterozygosity, distance/NJT, karyotyping, IGV, PLINK/VCF
export, SNP frequency analysis) - the "lower" branches from it, as opposed
to the "upper" (foundational) ones in the diagram above.

.. inheritance-diagram:: malariagen_data.anoph.pca.AnophelesPca malariagen_data.anoph.fst.AnophelesFstAnalysis malariagen_data.anoph.ld.AnophelesLdAnalysis malariagen_data.anoph.heterozygosity.AnophelesHetAnalysis malariagen_data.anoph.distance.AnophelesDistanceAnalysis malariagen_data.anoph.karyotype.AnophelesKaryotypeAnalysis malariagen_data.anoph.igv.AnophelesIgv malariagen_data.anoph.to_plink.PlinkConverter malariagen_data.anoph.to_vcf.SnpVcfExporter malariagen_data.anoph.snp_frq.AnophelesSnpFrequencyAnalysis malariagen_data.anoph.g123.AnophelesG123Analysis
    :parts: 1
    :top-classes: malariagen_data.anoph.snp_data.AnophelesSnpData, malariagen_data.anoph.hap_data.AnophelesHapData, malariagen_data.anoph.frq_base.AnophelesFrequencyAnalysis

Haplotype-based analyses
--------------------------
Everything built on :class:`~malariagen_data.anoph.hap_data.AnophelesHapData`
(selection scans H12/H1X/XP-EHH, haplotype clustering and frequency
analysis, diplotype clustering).

.. inheritance-diagram:: malariagen_data.anoph.hapclust.AnophelesHapClustAnalysis malariagen_data.anoph.h12.AnophelesH12Analysis malariagen_data.anoph.h1x.AnophelesH1XAnalysis malariagen_data.anoph.xpehh.AnophelesXpehhAnalysis malariagen_data.anoph.hap_frq.AnophelesHapFrequencyAnalysis malariagen_data.anoph.dipclust.AnophelesDipClustAnalysis
    :parts: 1
    :top-classes: malariagen_data.anoph.hap_data.AnophelesHapData, malariagen_data.anoph.snp_data.AnophelesSnpData, malariagen_data.anoph.snp_frq.AnophelesSnpFrequencyAnalysis, malariagen_data.anoph.frq_base.AnophelesFrequencyAnalysis, malariagen_data.anoph.cnv_frq.AnophelesCnvFrequencyAnalysis, malariagen_data.anoph.cnv_data.AnophelesCnvData

CNV-based analyses
-------------------
Everything built on :class:`~malariagen_data.anoph.cnv_data.AnophelesCnvData`
(gene CNV frequency analysis, and diplotype clustering, which also draws on
the SNP-frequency side shown in the previous diagram).

.. inheritance-diagram:: malariagen_data.anoph.cnv_frq.AnophelesCnvFrequencyAnalysis malariagen_data.anoph.dipclust.AnophelesDipClustAnalysis
    :parts: 1
    :top-classes: malariagen_data.anoph.cnv_data.AnophelesCnvData, malariagen_data.anoph.frq_base.AnophelesFrequencyAnalysis, malariagen_data.anoph.snp_frq.AnophelesSnpFrequencyAnalysis

Mixin class reference
-----------------------
The mixin classes shown above aren't individually documented anywhere
else, so each generates a full reference page, listing the methods it
contributes directly (not those it inherits).

.. autosummary::
    :toctree: generated/

    malariagen_data.anopheles.AnophelesDataResource
    malariagen_data.anoph.dipclust.AnophelesDipClustAnalysis
    malariagen_data.anoph.cnv_frq.AnophelesCnvFrequencyAnalysis
    malariagen_data.anoph.cnv_data.AnophelesCnvData
    malariagen_data.anoph.hapclust.AnophelesHapClustAnalysis
    malariagen_data.anoph.snp_frq.AnophelesSnpFrequencyAnalysis
    malariagen_data.anoph.xpehh.AnophelesXpehhAnalysis
    malariagen_data.anoph.h1x.AnophelesH1XAnalysis
    malariagen_data.anoph.h12.AnophelesH12Analysis
    malariagen_data.anoph.g123.AnophelesG123Analysis
    malariagen_data.anoph.fst.AnophelesFstAnalysis
    malariagen_data.anoph.heterozygosity.AnophelesHetAnalysis
    malariagen_data.anoph.hap_frq.AnophelesHapFrequencyAnalysis
    malariagen_data.anoph.distance.AnophelesDistanceAnalysis
    malariagen_data.anoph.pca.AnophelesPca
    malariagen_data.anoph.to_plink.PlinkConverter
    malariagen_data.anoph.ld.AnophelesLdAnalysis
    malariagen_data.anoph.to_vcf.SnpVcfExporter
    malariagen_data.anoph.igv.AnophelesIgv
    malariagen_data.anoph.karyotype.AnophelesKaryotypeAnalysis
    malariagen_data.anoph.aim_data.AnophelesAimData
    malariagen_data.anoph.hap_data.AnophelesHapData
    malariagen_data.anoph.snp_data.AnophelesSnpData
    malariagen_data.anoph.sample_metadata.AnophelesSampleMetadata
    malariagen_data.anoph.genome_features.AnophelesGenomeFeaturesData
    malariagen_data.anoph.genome_sequence.AnophelesGenomeSequenceData
    malariagen_data.anoph.frq_base.AnophelesFrequencyAnalysis
    malariagen_data.anoph.describe.AnophelesDescribe
    malariagen_data.anoph.base.AnophelesBase
    malariagen_data.anoph.phenotypes.AnophelesPhenotypeData
