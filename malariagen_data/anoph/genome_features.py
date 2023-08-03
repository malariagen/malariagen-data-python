from typing import Dict, Optional, Tuple

import bokeh.models
import bokeh.plotting
import numpy as np
import pandas as pd
from numpydoc_decorator import doc
from pandas.io.common import infer_compression

from ..util import (
    Region,
    check_types,
    parse_multi_region,
    parse_single_region,
    read_gff3,
    unpack_gff3_attributes,
)
from . import base_params, gplt_params
from .base_params import DEFAULT
from .genome_sequence import AnophelesGenomeSequenceData


class AnophelesGenomeFeaturesData(AnophelesGenomeSequenceData):
    def __init__(
        self,
        *,
        gff_gene_type: str,
        gff_default_attributes: Tuple[str, ...],
        **kwargs,
    ):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

        # TODO Consider moving these parameters to configuration, as they could
        # change if the GFF ever changed.
        self._gff_gene_type = gff_gene_type
        self._gff_default_attributes = gff_default_attributes

        # Setup caches.
        self._cache_genome_features: Dict[Tuple[str, ...], pd.DataFrame] = dict()

    @property
    def _geneset_gff3_path(self):
        return self.config["GENESET_GFF3_PATH"]

    def geneset(self, *args, **kwargs):
        """Deprecated, this method has been renamed to genome_features()."""
        return self.genome_features(*args, **kwargs)

    def _genome_features(self, *, attributes: Tuple[str, ...]):
        try:
            df = self._cache_genome_features[attributes]

        except KeyError:
            path = f"{self._base_path}/{self._geneset_gff3_path}"
            compression = infer_compression(path, compression="infer")
            with self._fs.open(path, mode="rb") as f:
                df = read_gff3(f, compression=compression)
            if attributes:
                df = unpack_gff3_attributes(df, attributes=attributes)
            self._cache_genome_features[attributes] = df

        return df

    def _genome_features_for_contig(self, *, contig: str, attributes: Tuple[str, ...]):
        debug = self._log.debug

        df = self._genome_features(attributes=attributes)

        debug("Apply contig query.")
        return df.query(f"contig == '{contig}'")

    def _prep_gff_attributes(
        self, attributes: base_params.gff_attributes
    ) -> Tuple[str, ...]:
        if attributes is None:
            attributes_normed: Tuple[str, ...] = ()
        elif attributes == DEFAULT:
            attributes_normed = self._gff_default_attributes
        elif isinstance(attributes, str):
            attributes_normed = (attributes,)
        else:
            attributes_normed = tuple(attributes)
        return attributes_normed

    @check_types
    @doc(
        summary="Access genome feature annotations.",
        returns="A dataframe of genome annotations, one row per feature.",
    )
    def genome_features(
        self,
        region: Optional[base_params.regions] = None,
        attributes: base_params.gff_attributes = DEFAULT,
    ) -> pd.DataFrame:
        debug = self._log.debug

        attributes_normed = self._prep_gff_attributes(attributes)
        del attributes

        if region is not None:
            debug("Handle region.")
            regions = parse_multi_region(self, region)
            del region

            debug("Apply region query.")
            parts = []
            for r in regions:
                df_part = self._genome_features_for_contig(
                    contig=r.contig, attributes=attributes_normed
                )
                if r.end is not None:
                    df_part = df_part.query(f"start <= {r.end}")
                if r.start is not None:
                    df_part = df_part.query(f"end >= {r.start}")
                parts.append(df_part)
            df = pd.concat(parts, axis=0)
            return df.reset_index(drop=True).copy()

        return (
            self._genome_features(attributes=attributes_normed)
            .reset_index(drop=True)
            .copy()
        )

    def genome_feature_children(
        self, parent: str, attributes: base_params.gff_attributes = DEFAULT
    ) -> pd.DataFrame:
        # Normalise attributes and ensure Parent is included.
        attributes_normed = self._prep_gff_attributes(attributes)
        if "Parent" not in attributes_normed:
            attributes_normed += ("Parent",)

        # Obtain dataframe of all genome features.
        df_gf = self._genome_features(attributes=attributes_normed).copy()

        # Split the Parent column and explode.
        # See also https://github.com/malariagen/malariagen-data-python/issues/334
        df_gf["Parent"] = df_gf["Parent"].str.split(",")
        df_gf = df_gf.explode(column="Parent", ignore_index=True)

        # Query to find children of the requested parent.
        df_children = df_gf.query(f"Parent == '{parent}'")

        return df_children.copy()

    @check_types
    @doc(summary="Plot a transcript, using bokeh.")
    def plot_transcript(
        self,
        transcript: base_params.transcript,
        sizing_mode: gplt_params.sizing_mode = gplt_params.sizing_mode_default,
        width: gplt_params.width = gplt_params.width_default,
        height: gplt_params.height = 100,
        show: gplt_params.show = True,
        x_range: Optional[gplt_params.x_range] = None,
        toolbar_location: Optional[
            gplt_params.toolbar_location
        ] = gplt_params.toolbar_location_default,
        title: gplt_params.title = True,
    ) -> gplt_params.figure:
        debug = self._log.debug

        debug("Find the transcript annotation.")
        df_genome_features = self.genome_features().set_index("ID")
        parent = df_genome_features.loc[transcript]

        if title is True:
            title = f"{transcript} ({parent.strand})"

        if x_range is None:
            x_range = bokeh.models.Range1d(
                parent.start - 2_000, parent.end + 2_000, bounds="auto"
            )

        debug("Define tooltips for hover.")
        tooltips = [
            ("Type", "@type"),
            ("Location", "@contig:@start{,}-@end{,}"),
        ]

        debug("Make a figure.")
        xwheel_zoom = bokeh.models.WheelZoomTool(
            dimensions="width", maintain_focus=False
        )
        fig = bokeh.plotting.figure(
            title=title,
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            tools=["xpan", "xzoom_in", "xzoom_out", xwheel_zoom, "reset", "hover"],
            toolbar_location=toolbar_location,
            active_scroll=xwheel_zoom,
            active_drag="xpan",
            tooltips=tooltips,
            x_range=x_range,
            y_range=bokeh.models.Range1d(-0.6, 0.6),
        )

        debug("Find child components of the transcript.")
        data = self.genome_feature_children(parent=transcript, attributes=None)
        data["bottom"] = -0.4
        data["top"] = 0.4

        debug("Plot exons.")
        exons = data.query("type == 'exon'")
        fig.quad(
            bottom="bottom",
            top="top",
            left="start",
            right="end",
            source=exons,
            fill_color=None,
            line_color="black",
            line_width=0.5,
            fill_alpha=0,
        )

        debug("Plot introns.")
        for intron_start, intron_end in zip(exons[:-1]["end"], exons[1:]["start"]):
            intron_midpoint = (intron_start + intron_end) / 2
            line_data = pd.DataFrame(
                {
                    "x": [intron_start, intron_midpoint, intron_end],
                    "y": [0, 0.1, 0],
                    "type": "intron",
                    "contig": parent.contig,
                    "start": intron_start,
                    "end": intron_end,
                }
            )
            fig.line(
                x="x",
                y="y",
                source=line_data,
                line_width=1,
                line_color="black",
            )

        debug("Plot UTRs.")
        fig.quad(
            bottom="bottom",
            top="top",
            left="start",
            right="end",
            source=data.query("type == 'five_prime_UTR'"),
            fill_color="green",
            line_width=0,
            fill_alpha=0.5,
        )
        fig.quad(
            bottom="bottom",
            top="top",
            left="start",
            right="end",
            source=data.query("type == 'three_prime_UTR'"),
            fill_color="red",
            line_width=0,
            fill_alpha=0.5,
        )

        debug("Plot CDSs.")
        fig.quad(
            bottom="bottom",
            top="top",
            left="start",
            right="end",
            source=data.query("type == 'CDS'"),
            fill_color="blue",
            line_width=0,
            fill_alpha=0.5,
        )

        debug("Tidy up the figure.")
        fig.yaxis.ticker = []
        self._bokeh_style_genome_xaxis(fig, parent.contig)

        if show:  # pragma: no cover
            bokeh.plotting.show(fig)
            return None
        else:
            return fig

    @check_types
    @doc(
        summary="Plot a genes track, using bokeh.",
    )
    def plot_genes(
        self,
        region: base_params.region,
        sizing_mode: gplt_params.sizing_mode = gplt_params.sizing_mode_default,
        width: gplt_params.width = gplt_params.width_default,
        height: gplt_params.genes_height = 120,
        show: gplt_params.show = True,
        toolbar_location: Optional[
            gplt_params.toolbar_location
        ] = gplt_params.toolbar_location_default,
        x_range: Optional[gplt_params.x_range] = None,
        title: Optional[gplt_params.title] = None,
        output_backend: gplt_params.output_backend = gplt_params.output_backend_default,
    ) -> gplt_params.figure:
        debug = self._log.debug

        debug("handle region parameter - this determines the genome region to plot")
        resolved_region: Region = parse_single_region(self, region)
        del region

        debug("handle region bounds")
        contig = resolved_region.contig
        start = resolved_region.start
        end = resolved_region.end
        if start is None:
            start = 0
        if end is None:
            end = len(self.genome_sequence(contig))

        debug("define x axis range")
        if x_range is None:
            x_range = bokeh.models.Range1d(start, end, bounds="auto")

        debug("select the genes overlapping the requested region")
        data, tooltips = self._plot_genes_setup_data(region=resolved_region)

        debug(
            "we're going to plot each gene as a rectangle, so add some additional columns"
        )
        data["bottom"] = np.where(data["strand"] == "+", 1, 0)
        data["top"] = data["bottom"] + 0.8

        debug("tidy up missing values for presentation")
        data.fillna("", inplace=True)

        debug("make a figure")
        xwheel_zoom = bokeh.models.WheelZoomTool(
            dimensions="width", maintain_focus=False
        )
        fig = bokeh.plotting.figure(
            title=title,
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            tools=[
                "xpan",
                "xzoom_in",
                "xzoom_out",
                xwheel_zoom,
                "reset",
                "tap",
                "hover",
            ],
            toolbar_location=toolbar_location,
            active_scroll=xwheel_zoom,
            active_drag="xpan",
            tooltips=tooltips,
            x_range=x_range,
            y_range=bokeh.models.Range1d(-0.4, 2.2),
            output_backend=output_backend,
        )

        debug("add functionality to click through to vectorbase")
        url = "https://vectorbase.org/vectorbase/app/record/gene/@ID"
        taptool = fig.select(type=bokeh.models.TapTool)
        taptool.callback = bokeh.models.OpenURL(url=url)

        debug("now plot the genes as rectangles")
        fig.quad(
            bottom="bottom",
            top="top",
            left="start",
            right="end",
            source=data,
            line_width=0,
        )

        debug("tidy up the plot")
        fig.ygrid.visible = False
        yticks = [0.4, 1.4]
        yticklabels = ["-", "+"]
        fig.yaxis.ticker = yticks
        fig.yaxis.major_label_overrides = {k: v for k, v in zip(yticks, yticklabels)}
        fig.yaxis.axis_label = "Genes"
        self._bokeh_style_genome_xaxis(fig, contig)

        if show:  # pragma: no cover
            bokeh.plotting.show(fig)
            return None
        else:
            return fig

    def _plot_genes_setup_data(self, *, region):
        attributes = [a for a in self._gff_default_attributes if a != "Parent"]
        df_genome_features = self.genome_features(region=region, attributes=attributes)
        data = df_genome_features.query(f"type == '{self._gff_gene_type}'").copy()
        tooltips = [(a.capitalize(), f"@{a}") for a in attributes]
        tooltips += [("Location", "@contig:@start{,}-@end{,}")]
        return data, tooltips

    @staticmethod
    def _bokeh_style_genome_xaxis(fig, contig):
        """Standard styling for X axis of genome plots."""
        fig.xaxis.axis_label = f"Contig {contig} position (bp)"
        fig.xaxis.ticker = bokeh.models.AdaptiveTicker(min_interval=1)
        fig.xaxis.minor_tick_line_color = None
        fig.xaxis[0].formatter = bokeh.models.NumeralTickFormatter(format="0,0")
