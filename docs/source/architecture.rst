Package architecture
=====================

Each dataset class in ``malariagen_data`` (``Ag3``, ``Af1``, ``As1``,
``Amin1``, ``Adir1``, ``Adar1``) is a thin, dataset-specific subclass of
:class:`malariagen_data.AnophelesDataResource`, which itself is composed
from a number of mixin classes, each providing a related group of
functionality (e.g. SNP data access, PCA, selection scans, CNV data). All
six dataset classes share the exact same mixin structure below - only the
top-level dataset class and its configuration (which data are available,
default analyses, and so on) differ between them.

Click a box to jump to that class's entry below, which links on to its
source via a "[source]" link. This is a plain Graphviz-rendered SVG (the
same mechanism used on each dataset's own page, e.g. :doc:`Ag3`) rather than
an interactive JS diagram - it does not support panning/zooming beyond your
browser's own SVG zoom, but it renders identically and reliably in every
browser, which an earlier JS-based version here did not.

.. inheritance-diagram:: malariagen_data.ag3.Ag3 malariagen_data.af1.Af1 malariagen_data.as1.As1 malariagen_data.amin1.Amin1 malariagen_data.adir1.Adir1 malariagen_data.adar1.Adar1
    :parts: 1
    :top-classes: object

.. dropdown:: Class reference (click-through targets for the diagram above)

    These entries exist to give the diagram above somewhere to link to, and
    to provide a "[source]" link through to each class's code - they are not
    meant to be read top-to-bottom. For what each class actually provides,
    see :doc:`Ag3` (or the other dataset pages), which list the public
    methods grouped by topic.

    .. autoclass:: malariagen_data.ag3.Ag3
        :no-members:

    .. autoclass:: malariagen_data.af1.Af1
        :no-members:

    .. autoclass:: malariagen_data.as1.As1
        :no-members:

    .. autoclass:: malariagen_data.amin1.Amin1
        :no-members:

    .. autoclass:: malariagen_data.adir1.Adir1
        :no-members:

    .. autoclass:: malariagen_data.adar1.Adar1
        :no-members:

    .. autoclass:: malariagen_data.anopheles.AnophelesDataResource
        :no-members:

    .. autoclass:: malariagen_data.anoph.dipclust.AnophelesDipClustAnalysis
        :no-members:

    .. autoclass:: malariagen_data.anoph.cnv_frq.AnophelesCnvFrequencyAnalysis
        :no-members:

    .. autoclass:: malariagen_data.anoph.cnv_data.AnophelesCnvData
        :no-members:

    .. autoclass:: malariagen_data.anoph.hapclust.AnophelesHapClustAnalysis
        :no-members:

    .. autoclass:: malariagen_data.anoph.snp_frq.AnophelesSnpFrequencyAnalysis
        :no-members:

    .. autoclass:: malariagen_data.anoph.xpehh.AnophelesXpehhAnalysis
        :no-members:

    .. autoclass:: malariagen_data.anoph.h1x.AnophelesH1XAnalysis
        :no-members:

    .. autoclass:: malariagen_data.anoph.h12.AnophelesH12Analysis
        :no-members:

    .. autoclass:: malariagen_data.anoph.g123.AnophelesG123Analysis
        :no-members:

    .. autoclass:: malariagen_data.anoph.fst.AnophelesFstAnalysis
        :no-members:

    .. autoclass:: malariagen_data.anoph.heterozygosity.AnophelesHetAnalysis
        :no-members:

    .. autoclass:: malariagen_data.anoph.hap_frq.AnophelesHapFrequencyAnalysis
        :no-members:

    .. autoclass:: malariagen_data.anoph.distance.AnophelesDistanceAnalysis
        :no-members:

    .. autoclass:: malariagen_data.anoph.pca.AnophelesPca
        :no-members:

    .. autoclass:: malariagen_data.anoph.to_plink.PlinkConverter
        :no-members:

    .. autoclass:: malariagen_data.anoph.ld.AnophelesLdAnalysis
        :no-members:

    .. autoclass:: malariagen_data.anoph.to_vcf.SnpVcfExporter
        :no-members:

    .. autoclass:: malariagen_data.anoph.igv.AnophelesIgv
        :no-members:

    .. autoclass:: malariagen_data.anoph.karyotype.AnophelesKaryotypeAnalysis
        :no-members:

    .. autoclass:: malariagen_data.anoph.aim_data.AnophelesAimData
        :no-members:

    .. autoclass:: malariagen_data.anoph.hap_data.AnophelesHapData
        :no-members:

    .. autoclass:: malariagen_data.anoph.snp_data.AnophelesSnpData
        :no-members:

    .. autoclass:: malariagen_data.anoph.sample_metadata.AnophelesSampleMetadata
        :no-members:

    .. autoclass:: malariagen_data.anoph.genome_features.AnophelesGenomeFeaturesData
        :no-members:

    .. autoclass:: malariagen_data.anoph.genome_sequence.AnophelesGenomeSequenceData
        :no-members:

    .. autoclass:: malariagen_data.anoph.frq_base.AnophelesFrequencyAnalysis
        :no-members:

    .. autoclass:: malariagen_data.anoph.describe.AnophelesDescribe
        :no-members:

    .. autoclass:: malariagen_data.anoph.base.AnophelesBase
        :no-members:

    .. autoclass:: malariagen_data.anoph.phenotypes.AnophelesPhenotypeData
        :no-members:
