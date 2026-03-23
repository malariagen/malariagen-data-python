import json

import os
import dask.array as da
import pandas as pd
import xarray
import zarr
import ipyleaflet

from malariagen_data.util import (
    DIM_ALLELE,
    DIM_PLOIDY,
    DIM_SAMPLE,
    DIM_VARIANT,
    _prep_geneset_attributes_arg,
    _da_from_zarr,
    _init_filesystem,
    _init_zarr_store,
    _read_gff3,
    _resolve_region,
    _unpack_gff3_attributes,
)


class PlasmodiumDataResource:
    def __init__(
        self,
        data_config,
        url=None,
        **kwargs,
    ):
        # setup filesystem
        self.CONF = self._load_config(data_config)
        if not url:
            url = self.CONF["default_url"]
        self._fs, self._path = _init_filesystem(url, **kwargs)

        # setup caches
        self._cache_sample_metadata = None
        self._cache_variant_calls_zarr = None
        self._cache_genome = None
        self._cache_genome_features = dict()

        self.extended_calldata_variables = self.CONF["extended_calldata_variables"]
        self.extended_variant_fields = self.CONF["extended_variant_fields"]
        self.contigs = self.CONF["reference_contigs"]

    def _load_config(self, data_config):
        """Load the config for data structure on the cloud into json format."""
        with open(data_config) as json_conf:
            config_content = json.load(json_conf)
        return config_content

    def sample_metadata(
        self,
        sample_query=None,
        sample_query_options=None,
    ):
        """Access sample metadata and return as pandas dataframe.

        Parameters
        ----------
        sample_query : str or None, optional
            A pandas query string to filter samples, e.g.
            ``"QC pass == True"`` or ``"Country == 'Ghana'"``.
            Column names with spaces are supported without backticks.
            If None, all samples are returned.
        sample_query_options : dict or None, optional
            A dictionary of arguments passed through to pandas query(),
            e.g. ``parser``, ``engine``, ``local_dict``.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of sample metadata, one row per sample.
        """
        if self._cache_sample_metadata is None:
            path = os.path.join(self._path, self.CONF["metadata_path"])
            with self._fs.open(path) as f:
                self._cache_sample_metadata = pd.read_csv(f, sep="\t", na_values="")

        df = self._cache_sample_metadata

        if sample_query is not None:
            sample_query_options = sample_query_options or {}
            sample_query_options.setdefault("engine", "python")
            df = df.query(sample_query, **sample_query_options)
            df = df.reset_index(drop=True)

        return df.copy()

    def plot_samples_interactive_map(
        self,
        sample_query=None,
        sample_query_options=None,
        basemap="mapnik",
        center=(-2, 20),
        zoom=3,
        height=500,
        width="100%",
        min_samples=1,
        count_by="Population",
    ):
        """Plot an interactive map showing sampling locations using ipyleaflet.

        One marker is shown per unique first-level administrative division.
        Hovering over a marker shows the location name, country, years
        collected, and sample counts broken down by the ``count_by`` column
        (default ``"Population"``).

        Parameters
        ----------
        sample_query : str or None, optional
            A pandas query string to filter samples before plotting,
            e.g. ``"QC pass == True"`` or ``"Year > 2015"``.
            If None, all samples are shown.
        sample_query_options : dict or None, optional
            A dictionary of arguments passed through to pandas query(),
            e.g. ``parser``, ``engine``, ``local_dict``.
        basemap : str or dict or TileLayer or TileProvider, optional
            Basemap from ipyleaflet or other TileLayer provider. Strings are
            abbreviations mapped to corresponding basemaps, available values
            are ['mapnik', 'natgeoworldmap', 'opentopomap', 'positron',
            'satellite', 'worldimagery', 'worldstreetmap', 'worldtopomap'].
            Defaults to 'mapnik'.
        center : tuple of (int or float, int or float), optional
            Location to center the map. Defaults to (-2, 20).
        zoom : int or float, optional
            Initial zoom level. Defaults to 3.
        height : int or str, optional
            Height of the map in pixels or other CSS units. Defaults to 500.
        width : int or str, optional
            Width of the map in pixels or other CSS units. Defaults to '100%'.
        min_samples : int, optional
            Minimum number of samples required to show a marker for a given
            location. Defaults to 1.
        count_by : str, optional
            Metadata column to report counts of samples by for each location.
            Defaults to ``"Population"``.

        Returns
        -------
        ipyleaflet.Map
        """
        if isinstance(height, int):
            height = f"{height}px"
        if isinstance(width, int):
            width = f"{width}px"

        df = self.sample_metadata(
            sample_query=sample_query,
            sample_query_options=sample_query_options,
        )
        _column_candidates = {
            "admin_div": ["Admin level 1", "First-level administrative division"],
            "lat": ["Admin level 1 latitude", "Lat"],
            "lon": ["Admin level 1 longitude", "Long"],
        }

        def _resolve_col(key, candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            raise ValueError(
                f"Cannot find a column for {key!r}; tried {candidates}. "
                f"Available columns: {sorted(df.columns.tolist())}"
            )

        col_admin = _resolve_col("admin_div", _column_candidates["admin_div"])
        col_lat = _resolve_col("lat", _column_candidates["lat"])
        col_lon = _resolve_col("lon", _column_candidates["lon"])

        location_key = ["Country", col_admin, col_lat, col_lon]

        # Validate count_by column exists
        if count_by not in df.columns:
            raise ValueError(
                f"{count_by!r} is not a column in the sample metadata. "
                f"Available columns: {sorted(df.columns.tolist())}"
            )

        # Build pivot: one row per location, one column per count_by value
        df_pivot = df.pivot_table(
            index=location_key,
            columns=count_by,
            values="Sample",
            aggfunc="count",
            fill_value=0,
            observed=True,
        )
        df_pivot = df_pivot.reset_index()

        # Aggregate year and study info per location for tooltip
        df_aggs = (
            df.groupby(location_key)
            .agg(
                {
                    "Year": lambda x: ", ".join(
                        str(int(y)) for y in sorted(x.dropna().unique())
                    ),
                    "Study": lambda x: ", ".join(
                        str(s) for s in sorted(x.dropna().unique())
                    ),
                }
            )
            .reset_index()
        )

        df_pivot = df_pivot.merge(df_aggs, on=location_key)

        # Unique count_by values for building the tooltip breakdown
        count_factors = sorted(df[count_by].dropna().unique())

        # Basemap abbreviations — inlined from anoph/map_params.py
        _basemap_abbrev_candidates = {
            "mapnik": lambda: ipyleaflet.basemaps.OpenStreetMap.Mapnik,
            "natgeoworldmap": lambda: ipyleaflet.basemaps.Esri.NatGeoWorldMap,
            "opentopomap": lambda: ipyleaflet.basemaps.OpenTopoMap,
            "positron": lambda: ipyleaflet.basemaps.CartoDB.Positron,
            "satellite": lambda: ipyleaflet.basemaps.Gaode.Satellite,
            "worldimagery": lambda: ipyleaflet.basemaps.Esri.WorldImagery,
            "worldstreetmap": lambda: ipyleaflet.basemaps.Esri.WorldStreetMap,
            "worldtopomap": lambda: ipyleaflet.basemaps.Esri.WorldTopoMap,
        }

        # Resolve basemap — mirrors mosquito basemap handling exactly
        if isinstance(basemap, str):
            basemap_str = basemap.lower()
            if basemap_str not in _basemap_abbrev_candidates:
                raise ValueError(
                    f"Basemap abbreviation not recognised: {basemap_str!r}; "
                    f"try one of {list(_basemap_abbrev_candidates.keys())}"
                )
            try:
                basemap_provider = _basemap_abbrev_candidates[basemap_str]()
            except Exception:
                # Fall back to mapnik if provider is unavailable
                basemap_provider = ipyleaflet.basemaps.OpenStreetMap.Mapnik
        elif basemap is None:
            basemap_provider = ipyleaflet.basemaps.OpenStreetMap.Mapnik
        else:
            # User passed a TileLayer / TileProvider / dict directly
            basemap_provider = basemap

        # Create map
        samples_map = ipyleaflet.Map(
            center=center,
            zoom=zoom,
            basemap=basemap_provider,
        )
        samples_map.add(ipyleaflet.ScaleControl(position="bottomleft"))
        samples_map.layout.height = height
        samples_map.layout.width = width

        # Add one marker per location
        for _, row in df_pivot.iterrows():
            lat = row["Admin level 1 latitude"]
            lon = row["Admin level 1 longitude"]

            # Skip rows with missing coordinates
            if pd.isna(lat) or pd.isna(lon):
                continue

            # Build tooltip — mirrors mosquito API style
            title = f"Admin level 1: {row['Admin level 1']}"
            title += f"\nCountry: {row['Country']}"
            title += f"\nYears: {row['Year']}"
            title += f"\nStudies: {row['Study']}"

            total = 0
            breakdown = []
            for factor in count_factors:
                if factor in row:
                    n = row[factor]
                    total += n
                    if n > 0:
                        breakdown.append(f"{n} {factor}")

            # Respect min_samples threshold
            if total < min_samples:
                continue

            title += f"\nTotal samples: {total}"
            title += f"\nBy {count_by}: " + "; ".join(breakdown)

            marker = ipyleaflet.Marker(
                location=(lat, lon),
                draggable=False,
                title=title,
            )
            samples_map.add(marker)

        return samples_map

    def _open_variant_calls_zarr(self):
        """Open variant calls zarr.

        Returns
        -------
        root : zarr.hierarchy.Group
            Root of zarr containing information on variant calls.

        """
        if self._cache_variant_calls_zarr is None:
            path = os.path.join(self._path, self.CONF["variant_calls_zarr_path"])
            store = _init_zarr_store(fs=self._fs, path=path)
            self._cache_variant_calls_zarr = zarr.open_consolidated(store=store)
        return self._cache_variant_calls_zarr

    def _add_coordinates(self, root, inline_array, chunks, var_names_for_outputs):
        """Add coordinate variables in zarr to dictionary"""
        # coordinates
        coords = dict()
        for var_name in ["POS", "CHROM"]:
            z = root[f"variants/{var_name}"]
            var = _da_from_zarr(z, inline_array=inline_array, chunks=chunks)
            coords[f"variant_{var_names_for_outputs[var_name]}"] = [DIM_VARIANT], var

        z = root["samples"]
        sample_id = _da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        coords["sample_id"] = [DIM_SAMPLE], sample_id
        return coords

    def _add_default_data_vars(self, root, inline_array, chunks, var_names_for_outputs):
        """Add default set of variables from zarr to dictionary"""
        data_vars = dict()

        # Add variant_allele as combination of REF and ALT
        ref_z = root["variants/REF"]
        alt_z = root["variants/ALT"]
        ref = _da_from_zarr(ref_z, inline_array=inline_array, chunks=chunks)
        alt = _da_from_zarr(alt_z, inline_array=inline_array, chunks=chunks)
        variant_allele = da.concatenate([ref[:, None], alt], axis=1)
        data_vars["variant_allele"] = [DIM_VARIANT, DIM_ALLELE], variant_allele

        # other default variant values
        configurable_default_variant_variables = self.CONF["default_variant_variables"]
        for var_name in configurable_default_variant_variables:
            z = root[f"variants/{var_name}"]
            dimension = configurable_default_variant_variables[var_name]
            var = _da_from_zarr(z, inline_array=inline_array, chunks=chunks)
            if var_name in var_names_for_outputs.keys():
                var_name = var_names_for_outputs[var_name]
            data_vars[f"variant_{var_name}"] = dimension, var

        # call arrays
        gt_z = root["calldata/GT"]
        ad_z = root["calldata/AD"]
        call_genotype = _da_from_zarr(gt_z, inline_array=inline_array, chunks=chunks)
        call_ad = _da_from_zarr(ad_z, inline_array=inline_array, chunks=chunks)
        data_vars["call_genotype"] = (
            [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY],
            call_genotype,
        )
        data_vars["call_AD"] = ([DIM_VARIANT, DIM_SAMPLE, DIM_ALLELE], call_ad)

        return data_vars

    def _add_extended_data(self, root, inline_array, chunks, data_vars):
        """Add all variables not included in default set to dictionary"""

        subset_extended_variants = self.extended_variant_fields
        subset_extended_calldata = self.extended_calldata_variables

        for var_name in subset_extended_calldata:
            z = root[f"calldata/{var_name}"]
            var = _da_from_zarr(z, inline_array=inline_array, chunks=chunks)
            data_vars[f"call_{var_name}"] = (
                subset_extended_calldata[var_name],
                var,
            )

        for var_name in subset_extended_variants:
            z = root[f"variants/{var_name}"]
            field = _da_from_zarr(z, inline_array=inline_array, chunks=chunks)
            data_vars[f"variant_{var_name}"] = (
                subset_extended_variants[var_name],
                field,
            )
        return data_vars

    def variant_calls(self, extended=False, inline_array=True, chunks="native"):
        """Access variant sites, site filters and genotype calls.

        Parameters
        ----------
        extended : bool, optional
            If False only the default variables are returned. If True all variables from the zarr are returned. Defaults to False.
        inline_array : bool, optional
            Passed through to dask.array.from_array(). Defaults to True.
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr chunks.
            Also can be a target size, e.g., '200 MiB'. Defaults to "native".

        Returns
        -------
        ds : xarray.Dataset
            Dataset containing either default or extended variables from the variant calls Zarr.
        """
        # setup
        root = self._open_variant_calls_zarr()
        var_names_for_outputs = {
            "POS": "position",
            "CHROM": "chrom",
            "FILTER_PASS": "filter_pass",
        }

        # Add default data
        coords = self._add_coordinates(
            root, inline_array, chunks, var_names_for_outputs
        )
        data_vars = self._add_default_data_vars(
            root, inline_array, chunks, var_names_for_outputs
        )

        # Add extended data
        if extended:
            data_vars = self._add_extended_data(root, inline_array, chunks, data_vars)

        # create a dataset
        ds = xarray.Dataset(data_vars=data_vars, coords=coords)

        return ds

    def open_genome(self):
        """Open the reference genome zarr.

        Returns
        -------
        root : zarr.hierarchy.Group
            Zarr hierarchy containing the reference genome sequence.

        """
        if self._cache_genome is None:
            path = os.path.join(self._path, self.CONF["reference_path"])
            store = _init_zarr_store(fs=self._fs, path=path)
            self._cache_genome = zarr.open_consolidated(store=store)
        return self._cache_genome

    def _resolve_region(self, region):
        """Convert a genome region into a standard data structure.

        Parameters
        ----------
        region: str
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").

        Returns
        -------
        out : Region
            A named tuple with attributes contig, start and end.

        """

        return _resolve_region(self, region)

    def _subset_genome_sequence_region(
        self, genome, region, inline_array=True, chunks="native"
    ):
        """Sebset reference genome sequence."""
        region = self._resolve_region(region)
        z = genome[region.contig]

        d = _da_from_zarr(z, inline_array=inline_array, chunks=chunks)

        if region.start:
            slice_start = region.start - 1
        else:
            slice_start = None
        if region.end:
            slice_stop = region.end
        else:
            slice_stop = None
        loc_region = slice(slice_start, slice_stop)

        return d[loc_region]

    def genome_sequence(self, region="*", inline_array=True, chunks="native"):
        """Access the reference genome sequence.

        Parameters
        ----------
        region: str or list of str or Region or list of Region. Defaults to '*'
            Chromosome (e.g., "Pf3D7_07_v3"), gene name (e.g., "PF3D7_0709000"), genomic
            region defined with coordinates (e.g., "Pf3D7_07_v3:1-500").
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["Pf3D7_07_v3:1-500","Pf3D7_02_v3:15-20","Pf3D7_03_v3:40-50"].
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr
            chunks. Also, can be a target size, e.g., '200 MiB'.

        Returns
        -------
        d : dask.array.Array
            An array of nucleotides giving the reference genome sequence for the
            given region/gene/contig.

        """
        genome = self.open_genome()

        if region == "*" or region is None:
            regions = self.contigs
        elif isinstance(region, (list, tuple)):
            regions = list(region)
        else:
            regions = [region]

        sequences = [
            self._subset_genome_sequence_region(
                genome=genome,
                region=r,
                inline_array=inline_array,
                chunks=chunks,
            )
            for r in regions
        ]

        if len(sequences) == 1:
            return sequences[0]
        return da.concatenate(sequences)

    def genome_features(self, attributes=("ID", "Parent", "Name")):
        """Access genome feature annotations.

        Parameters
        ----------
        attributes : list of str, optional
            Attribute keys to unpack into columns. Provide "*" to unpack all attributes.

        Returns
        -------
        df : pandas.DataFrame

        """
        # Attributes
        attributes = _prep_geneset_attributes_arg(attributes)

        try:
            df = self._cache_genome_features[attributes]
        except KeyError:
            path = os.path.join(self._path, self.CONF["annotations_path"])
            with self._fs.open(path, mode="rb") as f:
                df = _read_gff3(f, compression="gzip")
            if attributes is not None:
                df = _unpack_gff3_attributes(df, attributes=attributes)
            self._cache_genome_features[attributes] = df

        return df
