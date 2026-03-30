"""Natural-language query parsing for malariagen_data API workflows.

The parser translates user questions into a structured representation
(:class:`ParsedQuery`) and then into API call specifications.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ParsedQuery:
    """Structured representation of a parsed natural language query."""

    task: str
    country: Optional[str] = None
    region: Optional[str] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    species: Optional[str] = None
    gene: Optional[str] = None
    populations: Optional[List[str]] = None
    visualization: Optional[str] = None
    sample_query: Optional[str] = None


class NLPQueryParser:
    """Parse natural language queries into malariagen_data API calls."""

    # Keyword mappings for different analysis tasks
    TASK_KEYWORDS: Dict[str, Tuple[str, ...]] = {
        "sample_metadata": [
            "samples",
            "find",
            "show",
            "filter",
            "which",
            "where",
            "how many",
            "list",
            "collection",
            "coordinates",
        ],
        "snp_frequencies": [
            "snp",
            "allele",
            "frequency",
            "frequencies",
            "variant",
            "polymorphism",
            "locus",
            "loci",
            "compare",
        ],
        "pca": [
            "pca",
            "principal component",
            "structure",
            "clustering",
            "population structure",
            "genetic structure",
        ],
        "fst": [
            "fst",
            "differentiation",
            "differentiated",
            "divergence",
            "selection",
            "gwss",
            "genome-wide",
            "distance between",
        ],
        "plot_map": [
            "map",
            "geographic",
            "coordinates",
            "spatial",
            "distribute",
            "plot",
            "visual",
        ],
    }

    TASK_PRIORITY: Tuple[str, ...] = (
        "fst",
        "pca",
        "snp_frequencies",
        "plot_map",
        "sample_metadata",
    )

    # Country/Region keywords
    COUNTRIES: Dict[str, str] = {
        "uganda": "Uganda",
        "benin": "Benin",
        "mali": "Mali",
        "cameroon": "Cameroon",
        "ghana": "Ghana",
        "guinea": "Guinea",
        "kenya": "Kenya",
        "tanzania": "Tanzania",
        "zambia": "Zambia",
        "mozambique": "Mozambique",
        "malawi": "Malawi",
        "senegal": "Senegal",
        "ivory coast": "Ivory Coast",
        "sierra leone": "Sierra Leone",
        "liberia": "Liberia",
        "west africa": "West Africa",
        "central africa": "Central Africa",
        "east africa": "East Africa",
        "africa": "Africa",
    }

    # Species keywords
    SPECIES: Dict[str, str] = {
        "gambiae": "gambiae",
        "coluzzii": "coluzzii",
        "arabiensis": "arabiensis",
        "merus": "merus",
        "funestus": "funestus",
        "nilus": "nilus",
    }

    # Genes of interest (especially resistance markers)
    GENES: Dict[str, str] = {
        "vgsc": "VGSC",
        "kdr": "KDR",
        "ace1": "ACE1",
        "gaba": "GABA",
        "cyp": "CYP",
    }

    VIZ_KEYWORDS: Dict[str, Tuple[str, ...]] = {
        "map": ("map", "geographic", "coordinates", "spatial"),
        "pca_plot": ("pca", "principal component", "structure"),
        "bar": ("bar", "histogram", "count", "distribution"),
        "heatmap": ("heatmap", "heat map"),
    }

    MIN_YEAR = 1900
    MAX_YEAR = 2100

    def __init__(self):
        """Initialize the parser."""
        self.query_history: List[str] = []

    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a natural language query into structured parameters.

        Parameters
        ----------
        query : str
            Natural language question about genomic data.

        Returns
        -------
        ParsedQuery
            Structured representation of the query with extracted entities.
        """
        query_lower = self._normalize_query(query)
        self.query_history.append(query)

        parsed = ParsedQuery(task="sample_metadata")

        parsed.task = self._identify_task(query_lower)

        locations = self._extract_locations(query_lower)
        country = locations[0] if locations else None
        if country:
            parsed.country = country

        year_range = self._extract_year(query_lower)
        if year_range:
            parsed.year_min, parsed.year_max = year_range

        species = self._extract_species(query_lower)
        if species:
            parsed.species = species

        gene = self._extract_gene(query_lower)
        if gene:
            parsed.gene = gene

        populations = self._extract_populations(locations)
        if populations:
            parsed.populations = populations

        visualization = self._extract_visualization(query_lower)
        if visualization:
            parsed.visualization = visualization

        parsed.sample_query = self._build_sample_query(
            country=country,
            year_min=year_range[0] if year_range else None,
            year_max=year_range[1] if year_range else None,
            species=species,
        )

        return parsed

    def _identify_task(self, query_lower: str) -> str:
        """Identify the main analytical task from keyword scoring."""
        scores: Dict[str, int] = {task: 0 for task in self.TASK_KEYWORDS}

        for task, keywords in self.TASK_KEYWORDS.items():
            for keyword in keywords:
                if self._contains_keyword(query_lower, keyword):
                    # Multi-word phrases are stronger signals than single words.
                    scores[task] += max(1, len(keyword.split()))

        best_score = max(scores.values())
        if best_score <= 0:
            return "sample_metadata"

        candidates = [task for task, score in scores.items() if score == best_score]
        for task in self.TASK_PRIORITY:
            if task in candidates:
                return task

        return "sample_metadata"

    def _extract_locations(self, query_lower: str) -> List[str]:
        """Extract countries/regions in mention order with de-duplication."""
        matches: List[Tuple[int, str]] = []

        for keyword, proper_name in self.COUNTRIES.items():
            for m in re.finditer(self._keyword_pattern(keyword), query_lower):
                matches.append((m.start(), proper_name))

        matches.sort(key=lambda x: x[0])

        ordered_unique: List[str] = []
        seen = set()
        for _, location in matches:
            if location not in seen:
                ordered_unique.append(location)
                seen.add(location)

        return ordered_unique

    def _extract_year(self, query_lower: str) -> Optional[Tuple[int, int]]:
        """Extract year range information."""
        # Pattern: "YEAR to YEAR" or "YEAR-YEAR"
        range_match = re.search(r"\b(\d{4})\s*(?:to|-)\s*(\d{4})\b", query_lower)
        if range_match:
            year_min = int(range_match.group(1))
            year_max = int(range_match.group(2))
            return self._normalize_year_range(year_min, year_max)

        # Pattern: "after YEAR", "since YEAR", or "from YEAR"
        after_match = re.search(r"\b(?:after|since|from)\s+(\d{4})\b", query_lower)
        if after_match:
            year = int(after_match.group(1))
            return self._normalize_year_range(year, dt.date.today().year)

        # Pattern: "before YEAR" or "until YEAR"
        before_match = re.search(r"\b(?:before|until)\s+(\d{4})\b", query_lower)
        if before_match:
            year = int(before_match.group(1))
            return self._normalize_year_range(self.MIN_YEAR, year)

        # Pattern: "in YEAR"
        exact_match = re.search(r"\bin\s+(\d{4})\b", query_lower)
        if exact_match:
            year = int(exact_match.group(1))
            return self._normalize_year_range(year, year)

        return None

    def _extract_species(self, query_lower: str) -> Optional[str]:
        """Extract species information."""
        for keyword, proper_name in self.SPECIES.items():
            if self._contains_keyword(query_lower, keyword):
                return proper_name
        return None

    def _extract_gene(self, query_lower: str) -> Optional[str]:
        """Extract gene/resistance marker information."""
        for keyword, proper_name in self.GENES.items():
            if self._contains_keyword(query_lower, keyword):
                return proper_name
        return None

    def _extract_populations(self, locations: List[str]) -> Optional[List[str]]:
        """Return extracted populations in deterministic mention order."""
        if not locations:
            return None
        return locations

    def _extract_visualization(self, query_lower: str) -> Optional[str]:
        """Identify requested visualization type."""
        for viz_type, keywords in self.VIZ_KEYWORDS.items():
            for keyword in keywords:
                if self._contains_keyword(query_lower, keyword):
                    return viz_type

        return None

    def _build_sample_query(
        self,
        country: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        species: Optional[str] = None,
    ) -> Optional[str]:
        """Build a sample_metadata query string."""
        conditions = []

        # Don't filter if the query asked for all Africa.
        if country and country != "Africa":
            conditions.append(f"country == '{country}'")

        if year_min is not None or year_max is not None:
            if year_min is not None and year_max is not None:
                conditions.append(f"(year >= {year_min}) & (year <= {year_max})")
            elif year_min is not None:
                conditions.append(f"year >= {year_min}")
            elif year_max is not None:
                conditions.append(f"year <= {year_max}")

        if species:
            conditions.append(f"species == '{species}'")

        if conditions:
            return " & ".join(conditions)

        return None

    def generate_api_call(self, parsed: ParsedQuery) -> Dict[str, Any]:
        """
        Convert a ParsedQuery into API call parameters.

        Parameters
        ----------
        parsed : ParsedQuery
            Parsed query object.

        Returns
        -------
        dict
            Parameters to pass to the appropriate API function.
        """
        api_call = {
            "function": parsed.task,
            "params": {},
        }

        # Common parameters
        if parsed.sample_query:
            api_call["params"]["sample_query"] = parsed.sample_query

        # Task-specific parameters
        if parsed.task == "sample_metadata":
            # Most common: just filter metadata
            pass

        elif parsed.task == "snp_frequencies":
            if parsed.populations:
                api_call["params"]["cohorts"] = parsed.populations
            else:
                api_call["params"]["region"] = "2L:1-100,000"  # Default region

        elif parsed.task == "pca":
            api_call["params"]["region"] = "2L:1-100,000"  # Default region
            api_call["params"]["n_snps"] = 5000

        elif parsed.task == "fst":
            if parsed.populations and len(parsed.populations) >= 2:
                api_call["params"]["cohorts"] = parsed.populations[:2]
            api_call["params"]["region"] = "genome"

        elif parsed.task == "plot_map":
            if parsed.visualization == "map":
                api_call["function"] = "plot_samples_interactive_map"

        return api_call

    def explain_query(self, parsed: ParsedQuery) -> str:
        """Generate a human-readable explanation of what was parsed."""
        lines = [f"Task: {parsed.task}"]

        if parsed.country:
            lines.append(f"  Location: {parsed.country}")

        if parsed.year_min or parsed.year_max:
            lines.append(f"  Years: {parsed.year_min or '?'} to {parsed.year_max or '?'}")

        if parsed.species:
            lines.append(f"  Species: {parsed.species}")

        if parsed.populations:
            lines.append(f"  Populations: {', '.join(parsed.populations)}")

        if parsed.gene:
            lines.append(f"  Gene of Interest: {parsed.gene}")

        if parsed.visualization:
            lines.append(f"  Visualization: {parsed.visualization}")

        return "\n".join(lines)

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join(query.lower().split())

    @staticmethod
    def _keyword_pattern(keyword: str) -> str:
        escaped = re.escape(keyword).replace(r"\ ", r"\s+")
        return rf"\b{escaped}\b"

    def _contains_keyword(self, text: str, keyword: str) -> bool:
        return re.search(self._keyword_pattern(keyword), text) is not None

    def _normalize_year_range(
        self,
        year_min: int,
        year_max: int,
    ) -> Optional[Tuple[int, int]]:
        if year_min > year_max:
            year_min, year_max = year_max, year_min

        if year_max < self.MIN_YEAR or year_min > self.MAX_YEAR:
            return None

        year_min = max(year_min, self.MIN_YEAR)
        year_max = min(year_max, self.MAX_YEAR)
        return year_min, year_max
