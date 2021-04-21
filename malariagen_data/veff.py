from __future__ import division, print_function

import collections
import operator

import petl as etl
from Bio.Seq import Seq


class Annotator(object):
    def __init__(self, genome, geneset):
        """
        An annotator.

        Parameters
        ----------

        genome : zarr hierarchy
            Reference genome.
        geneset : pandas dataframe
            Dataframe with genome annotations.

        """

        # store initialisation parameters
        self._genome = genome
        self._genome_cache = dict()
        # when debugging snp effects unhash seqid and add .eq("seqid", seqid) parameter to tbl_features
        # and seqid to __init__
        # self._seqid = seqid

        # setup access to GFF3 as a petl table
        # TODO at some point we'd like to refactor this module to read directly from pandas
        tbl_features = (
            etl.fromdataframe(geneset)
            .rename({"ID": "feature_id", "Parent": "parent_id", "end": "stop"})
            .select(lambda row: (row.stop - row.start) > 0)
        )
        self._tbl_features = tbl_features.cache()

        # index features by ID
        self._idx_feature_id = self._tbl_features.recordlookupone("feature_id")

        # index features by parent ID
        self._idx_parent_id = self._tbl_features.recordlookup("parent_id")

        # index features by genomic location
        self._idx_location = self._tbl_features.facetintervalrecordlookup(
            "seqid", "start", "stop", include_stop=True
        )

    def get_feature(self, feature_id):
        return self._idx_feature_id[feature_id]

    def get_children(self, feature_id):
        return self._idx_parent_id[feature_id]

    def find(self, chrom, start, stop):
        return self._idx_location[chrom].search(start, stop)

    def get_ref_seq(self, chrom, start, stop):
        """Accepts 1-based coords."""
        try:
            seq = self._genome_cache[chrom]
        except KeyError:
            seq = self._genome[chrom][:]
            self._genome_cache[chrom] = seq
        ref_seq = seq[start - 1 : stop]
        ref_seq = ref_seq.tobytes().decode()
        return ref_seq

    def get_ref_allele_coords(self, chrom, pos, ref):

        # N.B., use one-based inclusive coordinate system (like GFF3) throughout
        ref_start = pos
        ref_stop = pos + len(ref) - 1

        # check the reference allele matches the reference sequence
        ref_seq = self.get_ref_seq(chrom, ref_start, ref_stop).lower()
        assert ref_seq == ref.lower(), (
            "reference allele does not match reference sequence, "
            "expected %r, found %r" % (ref_seq, ref.lower())
        )

        return ref_start, ref_stop


VariantEffect = collections.namedtuple(
    "VariantEffect",
    (
        "effect",
        "impact",
        "chrom",
        "pos",
        "ref",
        "alt",
        "vlen",
        "ref_start",
        "ref_stop",
        "gene_id",
        "gene_start",
        "gene_stop",
        "gene_strand",
        "transcript_id",
        "transcript_start",
        "transcript_stop",
        "transcript_strand",
        "cds_id",
        "cds_start",
        "cds_stop",
        "cds_strand",
        "intron_start",
        "intron_stop",
        "intron_5prime_dist",
        "intron_3prime_dist",
        "intron_exon_5prime",
        "intron_exon_3prime",
        "ref_cds_start",
        "ref_cds_stop",
        "ref_intron_start",
        "ref_intron_stop",
        "ref_start_phase",
        "ref_codon",
        "alt_codon",
        "codon_change",
        "aa_pos",
        "ref_aa",
        "alt_aa",
        "aa_change",
    ),
)
null_effect = VariantEffect(*([None] * len(VariantEffect._fields)))


def get_effects(
    annotator,
    chrom,
    pos,
    ref,
    alt,
    gff3_gene_types={"gene", "pseudogene"},
    gff3_transcript_types={"mRNA", "rRNA", "pseudogenic_transcript"},
    gff3_cds_types={"CDS", "pseudogenic_exon"},
    transcript_ids=None,
):
    """TODO

    Parameters
    ----------

    gff3_gene_types : list of strings, optional
        Feature types to consider as genes.
    gff3_transcript_types : list of strings, optional
        Feature types to consider as transcripts.
    gff3_cds_types : list of strings, optional
        Feature types to consider as coding sequences.

    Returns
    -------

    A `VariantEffect` generator.

    """

    # ensure types and case
    ref = str(ref).upper()
    alt = str(alt).upper()

    # obtain start and stop coordinates of the reference allele
    ref_start, ref_stop = annotator.get_ref_allele_coords(chrom, pos, ref)

    # setup common effect parameters
    base_effect = null_effect._replace(
        chrom=chrom,
        pos=pos,
        ref=ref,
        alt=alt,
        vlen=len(alt) - len(ref),
        ref_start=ref_start,
        ref_stop=ref_stop,
    )

    # find overlapping genome features
    features = annotator.find(chrom, ref_start, ref_stop)

    # filter to find overlapping genes
    genes = [f for f in features if f.type in gff3_gene_types]

    if not genes:
        for effect in _get_intergenic_effects(annotator, base_effect):
            yield effect

    else:
        for gene in genes:
            for effect in _get_gene_effects(
                annotator,
                base_effect,
                gene,
                gff3_transcript_types,
                gff3_cds_types,
                transcript_ids,
            ):
                yield effect


# add as method
Annotator.get_effects = get_effects


def _get_intergenic_effects(annotator, base_effect):

    # TODO
    # UPSTREAM and DOWNSTREAM

    # the variant is in an intergenic region
    effect = base_effect._replace(effect="INTERGENIC", impact="MODIFIER")
    yield effect


def _get_gene_effects(
    annotator, base_effect, gene, gff3_transcript_types, gff3_cds_types, transcript_ids
):

    # setup common effect parameters
    base_effect = base_effect._replace(
        gene_id=gene.feature_id,
        gene_start=gene.start,
        gene_stop=gene.stop,
        gene_strand=gene.strand,
    )

    # obtain transcripts that are children of the current gene
    transcripts = [
        t
        for t in annotator.get_children(gene.feature_id)
        if t.type in gff3_transcript_types
    ]

    if not transcript_ids and not transcripts:

        # the variant hits a gene, but no transcripts within the gene
        effect = base_effect._replace(effect="INTRAGENIC", impact="MODIFIER")
        yield effect

    else:

        # optionally filter to user-specified transcripts
        if transcript_ids:
            transcripts = [t for t in transcripts if t.feature_id in transcript_ids]

        for transcript in transcripts:
            for effect in _get_transcript_effects(
                annotator, base_effect, transcript, gff3_cds_types
            ):
                yield effect


def _get_transcript_effects(annotator, base_effect, transcript, gff3_cds_types):

    # setup common effect parameters
    base_effect = base_effect._replace(
        transcript_id=transcript.feature_id,
        transcript_start=transcript.start,
        transcript_stop=transcript.stop,
        transcript_strand=transcript.strand,
    )

    # convenience
    ref_start = base_effect.ref_start
    ref_stop = base_effect.ref_stop
    transcript_start = transcript.start
    transcript_stop = transcript.stop

    # compare start and stop of reference allele to start
    # and stop of current transcript

    if ref_stop < transcript_start:

        # TODO
        # variant hits a gene but misses the current transcript, falling
        # upstream
        effect = base_effect._replace(effect="TODO")
        yield effect

    elif ref_start > transcript_stop:

        # TODO
        # variant hits a gene but misses the current transcript, falling
        # downstream
        effect = base_effect._replace(effect="TODO")
        yield effect

    elif ref_start < transcript_start <= ref_stop <= transcript_stop:

        # TODO
        # reference allele overhangs the start of the current transcript
        effect = base_effect._replace(effect="TODO")
        yield effect

    elif transcript_start <= ref_start <= transcript_stop < ref_stop:

        # TODO
        # reference allele overhangs the end of the current transcript
        effect = base_effect._replace(effect="TODO")
        yield effect

    elif ref_start < transcript_start <= transcript_stop < ref_stop:

        # TODO
        # reference allele entirely overlaps the current transcript and
        # overhangs at both ends
        effect = base_effect._replace(effect="TODO")
        yield effect

    else:

        # reference allele falls within current transcript
        assert transcript_start <= ref_start <= ref_stop <= transcript_stop
        for effect in _get_within_transcript_effects(
            annotator, base_effect, transcript, gff3_cds_types
        ):
            yield effect


def _get_within_transcript_effects(annotator, base_effect, transcript, gff3_cds_types):

    # convenience
    ref_start = base_effect.ref_start
    ref_stop = base_effect.ref_stop

    # obtain coding sequences that are children of the current transcript
    cdss = sorted(
        [
            f
            for f in annotator.get_children(transcript.feature_id)
            if f.type in gff3_cds_types
        ],
        key=lambda v: v.start,
    )

    exons = sorted(
        [f for f in annotator.get_children(transcript.feature_id) if f.type == "exon"],
        key=lambda v: v.start,
    )

    utr5 = sorted(
        [
            f
            for f in annotator.get_children(transcript.feature_id)
            if f.type == "five_prime_UTR"
        ],
        key=lambda v: v.start,
    )

    utr3 = sorted(
        [
            f
            for f in annotator.get_children(transcript.feature_id)
            if f.type == "three_prime_UTR"
        ],
        key=lambda v: v.start,
    )

    # derive introns, assuming between exons
    introns = [(x.stop + 1, y.start - 1) for x, y in zip(exons[:-1], exons[1:])]

    # introns_5utr = [(x.stop + 1, y.start - 1) for x, y in zip(utr5[:-1], utr5[1:])]
    #
    # # derive introns, assuming between CDSs
    # introns = [(x.stop + 1, y.start - 1) for x, y in zip(cdss[:-1], cdss[1:])]

    # if not cdss:
    #
    #     # TODO
    #     # the variant hits a transcript, but there are no CDSs within the
    #     # transcript
    #     effect = base_effect._replace(effect="TODO")
    #     yield effect

    # find coding sequence that overlaps the reference allele
    overlapping_cdss = [
        cds for cds in cdss if cds.start <= ref_stop and cds.stop >= ref_start
    ]

    overlapping_introns = [
        (start, stop)
        for (start, stop) in introns
        if start <= ref_stop and stop >= ref_start
    ]

    # overlapping_5utr_introns = [
    #     (start, stop)
    #     for (start, stop) in introns_5utr
    #     if start <= ref_stop and stop >= ref_start
    # ]

    overlapping_utr5 = [x for x in utr5 if x.start <= ref_stop and x.stop >= ref_start]

    overlapping_utr3 = [x for x in utr3 if x.start <= ref_stop and x.stop >= ref_start]

    # CDS effects

    if overlapping_cdss:

        if len(overlapping_cdss) > 1:

            # TODO
            # variant overlaps more than one exon
            effect = base_effect._replace(effect="TODO")
            yield effect

        else:

            # variant overlaps a single exon
            assert len(overlapping_cdss) == 1
            cds = overlapping_cdss[0]

            yield _get_cds_effect(annotator, base_effect, cds, cdss)

    # intron effects

    if overlapping_introns:

        if len(overlapping_introns) > 1:

            # TODO
            # variant overlaps more than one intron
            effect = base_effect._replace(effect="TODO")
            yield effect

        else:

            # variant overlaps a single intron
            assert len(overlapping_introns) == 1
            intron = overlapping_introns[0]

            yield _get_intron_effect(annotator, base_effect, intron, exons)

    # if overlapping_5utr_introns:
    #
    #     if len(overlapping_5utr_introns) > 1:
    #
    #         # TODO
    #         # variant overlaps more than one intron
    #         effect = base_effect._replace(effect="TODO")
    #         yield effect
    #
    #     else:
    #
    #         # variant overlaps a single intron in 5' utr
    #         assert len(overlapping_5utr_introns) == 1
    #         # TODO why doesnt this work
    #         # intron = overlapping_5utr_introns[0]
    #         # yield _get_intron_effect(annotator, base_effect, intron, cdss)
    #         effect = base_effect._replace(effect="INTRONIC")
    #         yield effect

    if overlapping_utr5:

        if len(overlapping_utr5) > 1:

            # TODO
            # variant overlaps more than one 5'UTR
            effect = base_effect._replace(effect="TODO")
            yield effect

        else:

            # variant overlaps a single 5 prime UTR
            assert len(overlapping_utr5) == 1
            utr5 = overlapping_utr5[0]

            effect = base_effect._replace(effect="FIVE_PRIME_UTR", impact="LOW")
            yield effect

    if overlapping_utr3:

        if len(overlapping_utr3) > 1:

            # TODO
            # variant overlaps more than one 3'UTR
            effect = base_effect._replace(effect="TODO")
            yield effect

        else:

            # variant overlaps a single 3 prime UTR
            assert len(overlapping_utr3) == 1
            utr3 = overlapping_utr3[0]

            effect = base_effect._replace(effect="THREE_PRIME_UTR", impact="LOW")
            yield effect

    # if none of the above - #
    if (
        (not overlapping_cdss)
        and (not overlapping_introns)
        and (not overlapping_utr5)
        and (not overlapping_utr3)
    ):
        effect = base_effect._replace(effect="INTRAGENIC", impact="LOW")
        yield effect


def _get_cds_effect(annotator, base_effect, cds, cdss):

    # setup common effect parameters
    base_effect = base_effect._replace(
        cds_id=cds.feature_id,
        cds_start=cds.start,
        cds_stop=cds.stop,
        cds_strand=cds.strand,
    )

    # convenience
    ref_start = base_effect.ref_start
    ref_stop = base_effect.ref_stop
    cds_start = cds.start
    cds_stop = cds.stop

    if ref_start < cds_start <= ref_stop <= cds_stop:

        # TODO
        # reference allele overhangs the start of the current exon
        effect = base_effect._replace(effect="TODO")
        return effect

    elif cds_start <= ref_start <= cds_stop < ref_stop:

        # TODO
        # reference allele overhangs the end of the current transcript
        effect = base_effect._replace(effect="TODO")
        return effect

    elif ref_start < cds_start <= cds_stop < ref_stop:

        # TODO
        # reference allele entirely overlaps the current exon and
        # overhangs at both ends
        effect = base_effect._replace(effect="TODO")
        return effect

    else:

        # reference allele falls within current transcript
        assert cds_start <= ref_start <= ref_stop <= cds_stop
        return _get_within_cds_effect(annotator, base_effect, cds, cdss)


def _get_within_cds_effect(annotator, base_effect, cds, cdss):

    # convenience
    chrom = base_effect.chrom
    pos = base_effect.pos
    ref = base_effect.ref
    alt = base_effect.alt
    strand = base_effect.cds_strand

    # obtain amino acid change
    (
        ref_cds_start,
        ref_cds_stop,
        ref_start_phase,
        ref_codon,
        alt_codon,
        aa_pos,
        ref_aa,
        alt_aa,
    ) = _get_aa_change(annotator, chrom, pos, ref, alt, cds, cdss)

    # setup common effect parameters
    base_effect = base_effect._replace(
        ref_cds_start=ref_cds_start,
        ref_cds_stop=ref_cds_stop,
        ref_start_phase=ref_start_phase,
        ref_codon=ref_codon,
        alt_codon=alt_codon,
        codon_change="%s/%s" % (ref_codon, alt_codon),
        aa_pos=aa_pos,
        ref_aa=ref_aa,
        alt_aa=alt_aa,
        aa_change="%s%s%s" % (ref_aa, aa_pos, alt_aa),
    )

    if len(ref) == 1 and len(alt) == 1:

        # SNPs

        if ref_aa == alt_aa:

            # TODO SYNONYMOUS_START and SYNONYMOUS_STOP

            # variant causes a codon that produces the same amino acid
            # e.g.: Ttg/Ctg, L/L
            effect = base_effect._replace(effect="SYNONYMOUS_CODING", impact="LOW")

        elif ref_aa == "M" and ref_cds_start == 0:

            # variant causes start codon to be mutated into a non-start codon.
            # e.g.: aTg/aGg, M/R
            effect = base_effect._replace(effect="START_LOST", impact="HIGH")

        elif ref_aa == "*":

            # variant causes stop codon to be mutated into a non-stop codon
            # e.g.: Tga/Cga, */R
            effect = base_effect._replace(effect="STOP_LOST", impact="HIGH")

        elif alt_aa == "*":

            # variant causes a STOP codon e.g.: Cag/Tag, Q/*
            effect = base_effect._replace(effect="STOP_GAINED", impact="HIGH")

        else:

            # TODO NON_SYNONYMOUS_START and NON_SYNONYMOUS_STOP

            # variant causes a codon that produces a different amino acid
            # e.g.: Tgg/Cgg, W/R
            effect = base_effect._replace(
                effect="NON_SYNONYMOUS_CODING", impact="MODERATE"
            )

    else:

        # INDELs and MNPs

        if (len(alt) - len(ref)) % 3:

            # N.B., this case covers both simple INDELs and complex
            # polymorphisms

            # insertion or deletion causes a frame shift
            # e.g.: An indel size is not multple of 3
            effect = base_effect._replace(effect="FRAME_SHIFT", impact="HIGH")

        elif len(ref) == 1 and len(ref) < len(alt):

            # simple insertions

            # figure out if there has been a codon change or not
            is_codon_changed = (strand == "+" and ref_aa[0] != alt_aa[0]) or (
                strand == "-" and ref_aa[-1] != alt_aa[-1]
            )

            if is_codon_changed:

                # one codon is changed and one or many codons are inserted
                # e.g.: An insert of size multiple of three, not at codon
                # boundary
                effect = base_effect._replace(
                    effect="CODON_CHANGE_PLUS_CODON_INSERTION", impact="MODERATE"
                )

            else:

                # one or many codons are inserted
                # e.g.: An insert multiple of three in a codon boundary
                effect = base_effect._replace(
                    effect="CODON_INSERTION", impact="MODERATE"
                )

        elif len(alt) == 1 and len(ref) > len(alt):

            # simple deletions

            # figure out if there has been a codon change or not
            is_codon_changed = (strand == "+" and ref_aa[0] != alt_aa[0]) or (
                strand == "-" and ref_aa[-1] != alt_aa[-1]
            )

            if is_codon_changed:

                # one codon is changed and one or many codons are deleted
                # e.g.: A deletion of size multiple of three, not at codon
                # boundary
                effect = base_effect._replace(
                    effect="CODON_CHANGE_PLUS_CODON_DELETION", impact="MODERATE"
                )

            else:

                # one or many codons are deleted
                # e.g.: A deletions multiple of three in a codon boundary
                effect = base_effect._replace(
                    effect="CODON_DELETION", impact="MODERATE"
                )

        elif len(ref) == len(alt):

            # MNPs
            effect = base_effect._replace(effect="CODON_CHANGE", impact="MODERATE")

        else:

            # TODO in-frame complex variation (MNP + INDEL)
            effect = base_effect._replace(effect="TODO")

    return effect


def _get_aa_change(annotator, chrom, pos, ref, alt, cds, cdss):

    # obtain codon change
    (
        ref_cds_start,
        ref_cds_stop,
        ref_start_phase,
        ref_codon,
        alt_codon,
    ) = _get_codon_change(annotator, chrom, pos, ref, alt, cds, cdss)

    # translate codon change to amino acid change
    ref_aa = str(Seq(ref_codon).translate())
    alt_aa = str(Seq(alt_codon).translate())
    aa_pos = (ref_cds_start // 3) + 1

    return (
        ref_cds_start,
        ref_cds_stop,
        ref_start_phase,
        ref_codon,
        alt_codon,
        aa_pos,
        ref_aa,
        alt_aa,
    )


def _get_codon_change(annotator, chrom, pos, ref, alt, cds, cdss):

    # obtain reference allele coords relative to coding sequence
    ref_start, ref_stop = annotator.get_ref_allele_coords(chrom, pos, ref)
    ref_cds_start, ref_cds_stop = _get_coding_position(ref_start, ref_stop, cds, cdss)

    # calculate position of reference allele start within codon
    ref_start_phase = ref_cds_start % 3

    if cds.strand == "+":

        # obtain any previous nucleotides to complete the first codon
        prefix = annotator.get_ref_seq(
            chrom=chrom, start=ref_start - ref_start_phase, stop=ref_start - 1
        ).lower()

        # begin constructing reference and alternate codon sequences
        ref_codon = prefix + ref
        alt_codon = prefix + alt

        # obtain any subsequence nucleotides to complete the last codon
        if len(ref_codon) % 3:
            ref_stop_phase = len(ref_codon) % 3
            suffix = annotator.get_ref_seq(
                chrom=chrom, start=ref_stop + 1, stop=ref_stop + 3 - ref_stop_phase
            )
            suffix = str(suffix).lower()
            ref_codon += suffix

        if len(alt_codon) % 3:
            alt_stop_phase = len(alt_codon) % 3
            suffix = annotator.get_ref_seq(
                chrom=chrom, start=ref_stop + 1, stop=ref_stop + 3 - alt_stop_phase
            ).lower()
            alt_codon += suffix

    else:

        # N.B., we are on the reverse strand, so position reported for
        # variant is actually position at the *end* of the reference allele
        # which is particularly important for deletions

        # we will construct everything for the forward strand (i.e., back-to-
        # front) then take reverse complement afterwards at the end of this
        # code block

        # obtain any previous nucleotides to complete the first codon
        prefix = annotator.get_ref_seq(
            chrom=chrom, start=ref_stop + 1, stop=ref_stop + ref_start_phase
        ).lower()

        # begin constructing reference and alternate codon sequences
        ref_codon = ref + prefix
        alt_codon = alt + prefix

        # obtain any subsequence nucleotides to complete the last codon
        if len(ref_codon) % 3:
            ref_stop_phase = len(ref_codon) % 3
            suffix = annotator.get_ref_seq(
                chrom=chrom, start=ref_start - 3 + ref_stop_phase, stop=ref_start - 1
            ).lower()
            ref_codon = suffix + ref_codon

        if len(alt_codon) % 3:
            alt_stop_phase = len(alt_codon) % 3
            suffix = annotator.get_ref_seq(
                chrom=chrom, start=ref_start - 3 + alt_stop_phase, stop=ref_start - 1
            ).lower()
            alt_codon = suffix + alt_codon

        # take reverse complement
        ref_codon = str(Seq(ref_codon).reverse_complement())
        alt_codon = str(Seq(alt_codon).reverse_complement())

    return ref_cds_start, ref_cds_stop, ref_start_phase, ref_codon, alt_codon


def _get_coding_position(ref_start, ref_stop, cds, cdss):
    # print('_get_coding_position', ref_start, ref_stop, cds, len(cdss))

    if cds.strand == "+":

        # sort exons
        cdss = sorted(cdss, key=operator.itemgetter("start"))

        # find index of overlapping exons in all exons
        cds_index = [f.start for f in cdss].index(cds.start)
        # print('_get_coding_position (+) cds_index', cds_index)

        # find offset
        offset = sum([f.stop - f.start + 1 for f in cdss[:cds_index]])
        # print('_get_coding_position (+) offset', offset)

        # find ref cds position
        ref_cds_start = offset + (ref_start - cds.start)
        ref_cds_stop = offset + (ref_stop - cds.start)

    else:

        # sort exons (backwards this time)
        cdss = sorted(cdss, key=operator.itemgetter("stop"), reverse=True)

        # find index of overlapping exons in all exons
        cds_index = [f.stop for f in cdss].index(cds.stop)
        # print('_get_coding_position (-) cds_index', cds_index)

        # find offset
        offset = sum([cds.stop - cds.start + 1 for cds in cdss[:cds_index]])
        # print('_get_coding_position (-) offset', offset)

        # find ref cds position
        ref_cds_start = offset + (cds.stop - ref_stop)
        ref_cds_stop = offset + (cds.stop - ref_start)

    # print('_get_coding_position return', ref_cds_start, ref_cds_stop)
    return ref_cds_start, ref_cds_stop


def _get_intron_effect(annotator, base_effect, intron, exons):

    # convenience
    ref_start = base_effect.ref_start
    ref_stop = base_effect.ref_stop
    intron_start, intron_stop = intron

    if ref_start < intron_start <= ref_stop <= intron_stop:

        # TODO
        # reference allele overhangs the start of the current intron
        effect = base_effect._replace(effect="TODO")
        return effect

    elif intron_start <= ref_start <= intron_stop < ref_stop:

        # TODO
        # reference allele overhangs the end of the current intron
        effect = base_effect._replace(effect="TODO")
        return effect

    elif ref_start < intron_start <= intron_stop < ref_stop:

        # TODO
        # reference allele entirely overlaps the current intron and
        # overhangs at both ends
        effect = base_effect._replace(effect="TODO")
        return effect

    else:

        # reference allele falls within current intron
        assert intron_start <= ref_start <= ref_stop <= intron_stop
        return _get_within_intron_effect(annotator, base_effect, intron, exons)


def _get_within_intron_effect(annotator, base_effect, intron, exons):

    # convenience
    ref_start = base_effect.ref_start
    ref_stop = base_effect.ref_stop
    ref = base_effect.ref
    alt = base_effect.alt
    intron_start, intron_stop = intron
    strand = base_effect.gene_strand
    if strand == "+":
        intron_5prime_dist = ref_start - (intron_start - 1)
        intron_3prime_dist = ref_stop - (intron_stop + 1)
        intron_exon_5prime = [
            exon.feature_id for exon in exons if exon.stop == intron_start - 1
        ][0]
        intron_exon_3prime = [
            exon.feature_id for exon in exons if exon.start == intron_stop + 1
        ][0]
    else:
        intron_5prime_dist = (intron_stop + 1) - ref_stop
        intron_3prime_dist = (intron_start - 1) - ref_start
        intron_exon_3prime = [
            exon.feature_id for exon in exons if exon.stop == intron_start - 1
        ][0]
        intron_exon_5prime = [
            exon.feature_id for exon in exons if exon.start == intron_stop + 1
        ][0]

    # setup common effect parameters
    base_effect = base_effect._replace(
        intron_start=intron_start,
        intron_stop=intron_stop,
        ref_intron_start=ref_start - intron_start,
        ref_intron_stop=ref_stop - intron_start,
        intron_5prime_dist=intron_5prime_dist,
        intron_3prime_dist=intron_3prime_dist,
        intron_exon_5prime=intron_exon_5prime,
        intron_exon_3prime=intron_exon_3prime,
    )

    intron_min_dist = min(intron_5prime_dist, -intron_3prime_dist)

    if len(ref) == 1 and len(alt) == 1:

        # SNPs

        if intron_min_dist <= 2:
            # splice site variation
            effect = base_effect._replace(effect="SPLICE_CORE", impact="HIGH")

        elif intron_min_dist <= 7:

            # splice site variation
            effect = base_effect._replace(effect="SPLICE_REGION", impact="MODERATE")

        else:

            # intron modifier
            effect = base_effect._replace(effect="INTRONIC", impact="MODIFIER")

    else:

        # TODO
        # INDELs and MNPs
        effect = base_effect._replace(effect="TODO")

    return effect
