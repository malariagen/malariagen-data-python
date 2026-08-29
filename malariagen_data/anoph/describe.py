import inspect
from typing import Optional

import pandas as pd
from numpydoc_decorator import doc  # type: ignore

from ..util import _check_types
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

    @_check_types
    @doc(
        summary="""
            Get detailed information about a specific API method, including
            its parameters and return type.
        """,
        parameters=dict(
            method_name="Name of a public API method.",
        ),
        returns="""
            A dataframe with one row per parameter, containing the parameter
            name, type, description, and default value.
        """,
    )
    def describe_method(
        self,
        method_name: str,
    ) -> pd.DataFrame:
        attr = getattr(self, method_name, None)
        if attr is None or not callable(attr):
            raise ValueError(
                f"No public method named {method_name!r}. "
                f"Use describe_api() to list available methods."
            )

        sig = inspect.signature(attr)
        docstring = inspect.getdoc(attr) or ""
        param_docs = self._parse_param_docs(docstring)

        params_info = []
        for pname, param in sig.parameters.items():
            ptype = ""
            if param.annotation is not inspect.Parameter.empty:
                ptype = inspect.formatannotation(param.annotation)

            default = ""
            if param.default is not inspect.Parameter.empty:
                default = repr(param.default)

            description = param_docs.get(pname, "")

            params_info.append(
                {
                    "parameter": pname,
                    "type": ptype,
                    "default": default,
                    "description": description,
                }
            )

        return pd.DataFrame(params_info)

    # Known numpy-style section headers (lowercase, no punctuation).
    _SECTION_HEADERS = frozenset(
        {
            "parameters",
            "returns",
            "raises",
            "notes",
            "examples",
            "see also",
            "warnings",
            "references",
            "yields",
            "receives",
            "other parameters",
        }
    )

    @classmethod
    def _is_section_header(cls, stripped: str) -> bool:
        """Check if a line is a numpy-style section header."""
        return stripped.rstrip(": ").lower() in cls._SECTION_HEADERS

    @staticmethod
    def _parse_param_docs(docstring: str) -> dict:
        """Parse parameter descriptions from a numpy-style docstring."""
        params = {}
        lines = docstring.splitlines()
        in_params = False
        current_param = None
        current_desc_lines: list[str] = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Detect the Parameters section.
            if not in_params and stripped.rstrip(": ").lower() == "parameters":
                in_params = True
                continue
            if in_params and stripped.startswith("---"):
                continue

            # Stop at the next section header.
            if in_params and AnophelesDescribe._is_section_header(stripped):
                if current_param:
                    params[current_param] = " ".join(current_desc_lines).strip()
                in_params = False
                continue

            if not in_params:
                continue

            # Detect a parameter line (name : type).
            if " : " in stripped and not line[0:1].isspace():
                # Save previous param.
                if current_param:
                    params[current_param] = " ".join(current_desc_lines).strip()
                # Strip leading * for *args/**kwargs so keys match signature names.
                current_param = stripped.split(" : ")[0].strip().lstrip("*")
                current_desc_lines = []
            elif current_param and stripped:
                current_desc_lines.append(stripped)

        # Save the last parameter (when Parameters is the final section).
        if current_param:
            params[current_param] = " ".join(current_desc_lines).strip()

        return params

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
