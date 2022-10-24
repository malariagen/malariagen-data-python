import collections
import operator

from Bio.Seq import Seq

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
        "strand",
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


class Annotator(object):
    def __init__(self, genome, genome_features):
        """
        An annotator.

        Parameters
        ----------
        genome : zarr hierarchy
            Reference genome.
        genome_features : pandas dataframe
            Dataframe with genome annotations.

        """

        # store initialisation parameters
        self._genome = genome
        self._genome_cache = dict()
        self._genome_features_cache = None

        genome_features = genome_features[
            (genome_features.end - genome_features.start) > 0
        ]
        self._genome_features_cache = genome_features

        # index features by ID
        self._idx_feature_id = self._genome_features_cache.set_index("ID")

        # index features by parent ID
        self._idx_parent_id = self._genome_features_cache.set_index("Parent")

    def get_feature(self, feature_id):
        return self._idx_feature_id.loc[feature_id]

    def get_children(self, feature_id):
        return self._idx_parent_id.loc[feature_id]

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

    def get_effects(self, transcript, variants, progress=None):

        children = self.get_children(transcript).sort_values("start")
        feature = self.get_feature(transcript)

        # make sure all alleles are uppercase
        variants.ref_allele = variants.ref_allele.str.upper()
        variants.alt_allele = variants.alt_allele.str.upper()

        # get transcript children
        cdss = list(children[children.type == "CDS"].itertuples())
        exons = list(children[children.type == "exon"].itertuples())
        utr5 = list(children[children.type == "five_prime_UTR"].itertuples())
        utr3 = list(children[children.type == "three_prime_UTR"].itertuples())
        introns = [(x.end + 1, y.start - 1) for x, y in zip(exons[:-1], exons[1:])]

        effect_values = []
        impact_values = []
        ref_codon_values = []
        alt_codon_values = []
        aa_pos_values = []
        ref_aa_values = []
        alt_aa_values = []
        aa_change_values = []

        feature_contig = feature.contig
        feature_start = feature.start
        feature_stop = feature.end
        feature_strand = feature.strand

        variant_iterator = variants.itertuples(index=True)
        if progress:
            variant_iterator = progress(
                variant_iterator, desc="Compute SNP effects", total=len(variants)
            )

        for row in variant_iterator:
            # some parameters
            chrom = feature_contig
            pos = row.position
            ref = row.ref_allele
            alt = row.alt_allele

            # obtain start and stop coordinates of the reference allele
            ref_start, ref_stop = self.get_ref_allele_coords(chrom, pos, ref)

            # set up the common effect parameters
            base_effect = null_effect._replace(
                chrom=chrom,
                pos=pos,
                ref=ref,
                alt=alt,
                vlen=len(alt) - len(ref),
                ref_start=ref_start,
                ref_stop=ref_stop,
                strand=feature_strand,
            )

            # reference allele falls within current transcript
            assert feature_start <= ref_start <= ref_stop <= feature_stop

            effect = _get_within_transcript_effect(
                ann=self,
                base_effect=base_effect,
                cdss=cdss,
                utr5=utr5,
                utr3=utr3,
                introns=introns,
            )

            effect_values.append(effect.effect)
            impact_values.append(effect.impact)
            ref_codon_values.append(effect.ref_codon)
            alt_codon_values.append(effect.alt_codon)
            aa_pos_values.append(effect.aa_pos)
            ref_aa_values.append(effect.ref_aa)
            alt_aa_values.append(effect.alt_aa)
            aa_change_values.append(effect.aa_change)

        variants["transcript"] = transcript
        variants["effect"] = effect_values
        variants["impact"] = impact_values
        variants["ref_codon"] = ref_codon_values
        variants["alt_codon"] = alt_codon_values
        variants["aa_pos"] = aa_pos_values
        variants["ref_aa"] = ref_aa_values
        variants["alt_aa"] = alt_aa_values
        variants["aa_change"] = aa_change_values

        return variants


def _get_within_transcript_effect(ann, base_effect, cdss, utr5, utr3, introns):
    # convenience
    ref_start = base_effect.ref_start
    ref_stop = base_effect.ref_stop

    # find coding sequence that overlaps the reference allele
    within_cdss = [
        cds for cds in cdss if cds.start <= ref_start and cds.end >= ref_stop
    ]
    if within_cdss:
        return _get_within_cds_effect(ann, base_effect, within_cdss[0], cdss)

    within_introns = [
        (start, stop)
        for (start, stop) in introns
        if start <= ref_start and stop >= ref_stop
    ]
    if within_introns:
        return _get_within_intron_effect(
            base_effect=base_effect, intron=within_introns[0]
        )

    within_utr5 = [x for x in utr5 if x.start <= ref_start and x.end >= ref_stop]
    if within_utr5:
        effect = base_effect._replace(effect="FIVE_PRIME_UTR", impact="LOW")
        return effect

    within_utr3 = [x for x in utr3 if x.start <= ref_start and x.end >= ref_stop]
    if within_utr3:
        effect = base_effect._replace(effect="THREE_PRIME_UTR", impact="LOW")
        return effect

    # if none of the above
    effect = base_effect._replace(effect="TODO", impact="UNKNOWN")
    return effect


def _get_cds_effect(ann, base_effect, cds, cdss):
    # setup common effect parameters
    base_effect = base_effect._replace(
        cds_id=cds.ID,
        cds_start=cds.start,
        cds_stop=cds.end,
        cds_strand=cds.strand,
    )

    # convenience
    ref_start = base_effect.ref_start
    ref_stop = base_effect.ref_stop
    cds_start = cds.start
    cds_stop = cds.end

    # reference allele falls within current transcript
    assert cds_start <= ref_start <= ref_stop <= cds_stop
    return _get_within_cds_effect(ann, base_effect, cds, cdss)


def _get_within_cds_effect(ann, base_effect, cds, cdss):
    # convenience
    chrom = base_effect.chrom
    pos = base_effect.pos
    ref = base_effect.ref
    alt = base_effect.alt
    strand = base_effect.strand

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
    ) = _get_aa_change(ann, chrom, pos, ref, alt, cds, cdss)

    # setup common effect parameters
    base_effect = base_effect._replace(
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
            effect = base_effect._replace(effect="TODO", impact="UNKNOWN")

    return effect


def _get_aa_change(ann, chrom, pos, ref, alt, cds, cdss):
    # obtain codon change
    (
        ref_cds_start,
        ref_cds_stop,
        ref_start_phase,
        ref_codon,
        alt_codon,
    ) = _get_codon_change(ann, chrom, pos, ref, alt, cds, cdss)

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


def _get_codon_change(ann, chrom, pos, ref, alt, cds, cdss):
    # obtain reference allele coords relative to coding sequence
    ref_start, ref_stop = ann.get_ref_allele_coords(chrom, pos, ref)
    ref_cds_start, ref_cds_stop = _get_coding_position(ref_start, ref_stop, cds, cdss)

    # calculate position of reference allele start within codon
    ref_start_phase = ref_cds_start % 3

    if cds.strand == "+":

        # obtain any previous nucleotides to complete the first codon
        prefix = ann.get_ref_seq(
            chrom=chrom, start=ref_start - ref_start_phase, stop=ref_start - 1
        ).lower()

        # begin constructing reference and alternate codon sequences
        ref_codon = prefix + ref
        alt_codon = prefix + alt

        # obtain any subsequence nucleotides to complete the last codon
        if len(ref_codon) % 3:
            ref_stop_phase = len(ref_codon) % 3
            suffix = ann.get_ref_seq(
                chrom=chrom, start=ref_stop + 1, stop=ref_stop + 3 - ref_stop_phase
            )
            suffix = str(suffix).lower()
            ref_codon += suffix

        if len(alt_codon) % 3:
            alt_stop_phase = len(alt_codon) % 3
            suffix = ann.get_ref_seq(
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
        prefix = ann.get_ref_seq(
            chrom=chrom, start=ref_stop + 1, stop=ref_stop + ref_start_phase
        ).lower()

        # begin constructing reference and alternate codon sequences
        ref_codon = ref + prefix
        alt_codon = alt + prefix

        # obtain any subsequence nucleotides to complete the last codon
        if len(ref_codon) % 3:
            ref_stop_phase = len(ref_codon) % 3
            suffix = ann.get_ref_seq(
                chrom=chrom, start=ref_start - 3 + ref_stop_phase, stop=ref_start - 1
            ).lower()
            ref_codon = suffix + ref_codon

        if len(alt_codon) % 3:
            alt_stop_phase = len(alt_codon) % 3
            suffix = ann.get_ref_seq(
                chrom=chrom, start=ref_start - 3 + alt_stop_phase, stop=ref_start - 1
            ).lower()
            alt_codon = suffix + alt_codon

        # take reverse complement
        ref_codon = str(Seq(ref_codon).reverse_complement())
        alt_codon = str(Seq(alt_codon).reverse_complement())

    return ref_cds_start, ref_cds_stop, ref_start_phase, ref_codon, alt_codon


def _get_coding_position(ref_start, ref_stop, cds, cdss):
    if cds.strand == "+":

        # sort exons
        cdss = sorted(cdss, key=operator.attrgetter("start"))

        # find index of overlapping exons in all exons
        cds_index = [f.start for f in cdss].index(cds.start)

        # find offset
        offset = sum([f.end - f.start + 1 for f in cdss[:cds_index]])

        # find ref cds position
        ref_cds_start = offset + (ref_start - cds.start)
        ref_cds_stop = offset + (ref_stop - cds.start)

    else:

        # sort exons (backwards this time)
        cdss = sorted(cdss, key=operator.attrgetter("end"), reverse=True)

        # find index of overlapping exons in all exons
        cds_index = [f.end for f in cdss].index(cds.end)

        # find offset
        offset = sum([cds.end - cds.start + 1 for cds in cdss[:cds_index]])

        # find ref cds position
        ref_cds_start = offset + (cds.end - ref_stop)
        ref_cds_stop = offset + (cds.end - ref_start)

    return ref_cds_start, ref_cds_stop


def _get_within_intron_effect(base_effect, intron):
    # convenience
    ref_start = base_effect.ref_start
    ref_stop = base_effect.ref_stop
    ref = base_effect.ref
    alt = base_effect.alt
    intron_start, intron_stop = intron
    strand = base_effect.strand
    if strand == "+":
        intron_5prime_dist = ref_start - (intron_start - 1)
        intron_3prime_dist = ref_stop - (intron_stop + 1)

    else:
        intron_5prime_dist = (intron_stop + 1) - ref_stop
        intron_3prime_dist = (intron_start - 1) - ref_start

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

        # TODO INDELs and MNPs
        effect = base_effect._replace(effect="TODO")

    return effect
