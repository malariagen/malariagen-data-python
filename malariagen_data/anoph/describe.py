import inspect
from typing import Optional

import pandas as pd
from numpydoc_decorator import doc  # type: ignore

from .base import AnophelesBase


class AnophelesDescribe(AnophelesBase):
    """Mixin class providing API introspection and discovery functionality."""

    @doc(
        summary="""
            List all available public API methods with their descriptions.
        """,
        returns="""
            A dataframe with one row per public method, containing the method
            name, a short summary description, and its category (data access,
            analysis, or plotting).
        """,
        parameters=dict(
            category="""
                Optional filter to show only methods of a given category.
                Supported values are "data", "analysis", "plot", or None to
                show all methods.
            """,
        ),
    )
    def describe_api(
        self,
        category: Optional[str] = None,
    ) -> pd.DataFrame:
        methods_info = []

        # Walk through all public methods on this instance.
        for name in sorted(dir(self)):
            # Skip private/dunder methods.
            if name.startswith("_"):
                continue

            attr = getattr(type(self), name, None)
            if attr is None:
                continue

            # Only include callable methods and non-property attributes.
            if isinstance(attr, property):
                continue
            if not callable(attr):
                continue

            # Extract the docstring summary.
            summary = self._extract_summary(attr)

            # Determine category.
            method_category = self._categorize_method(name)

            methods_info.append(
                {
                    "method": name,
                    "summary": summary,
                    "category": method_category,
                }
            )

        df = pd.DataFrame(methods_info)

        # Apply category filter if specified.
        if category is not None:
            valid_categories = {"data", "analysis", "plot"}
            if category not in valid_categories:
                raise ValueError(
                    f"Invalid category: {category!r}. "
                    f"Must be one of {valid_categories}."
                )
            df = df[df["category"] == category].reset_index(drop=True)

        return df

    @staticmethod
    def _extract_summary(method) -> str:
        """Extract the first line of the docstring as a summary."""
        docstring = inspect.getdoc(method)
        if not docstring:
            return ""
        # Take the first non-empty line as the summary.
        for line in docstring.strip().splitlines():
            line = line.strip()
            if line:
                return line
        return ""

    @staticmethod
    def _categorize_method(name: str) -> str:
        """Categorize a method based on its name."""
        if name.startswith("plot_"):
            return "plot"
        data_prefixes = (
            "sample_",
            "snp_",
            "hap_",
            "cnv_",
            "genome_",
            "open_",
            "lookup_",
            "read_",
            "general_",
            "sequence_",
            "cohorts_",
            "aim_",
            "gene_",
        )
        if name.startswith(data_prefixes):
            return "data"
        return "analysis"
