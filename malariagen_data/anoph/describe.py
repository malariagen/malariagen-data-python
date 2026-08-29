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

    @doc(
        summary="""
            Describe the parameters of a public API method.
        """,
        returns="""
            A dataframe with one row per parameter, containing the parameter
            name, type, default value, and description.
        """,
        parameters=dict(
            method="""
                Name of the public API method to describe.
            """,
        ),
    )
    def describe_method(self, method: str) -> pd.DataFrame:
        method_name = method

        # Reject private/dunder names.
        if method_name.startswith("_"):
            raise ValueError(f"Private method not allowed: {method_name!r}")

        # Method must exist on this instance.
        if not hasattr(self, method_name):
            raise ValueError(f"Unknown method: {method_name!r}")

        attr = getattr(type(self), method_name, None)
        if attr is None:
            raise ValueError(f"Unknown method: {method_name!r}")

        # Must be callable and not a property.
        if isinstance(attr, property) or not callable(attr):
            raise ValueError(f"Method is not callable: {method_name!r}")

        sig = inspect.signature(attr)
        param_docs = self._extract_parameter_docs(attr)

        rows = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            annotation = (
                None
                if param.annotation is inspect.Signature.empty
                else str(param.annotation)
            )
            default = (
                None if param.default is inspect.Signature.empty else param.default
            )

            doc_info = param_docs.get(param_name, {})
            doc_type = doc_info.get("type")
            description = doc_info.get("description")

            param_type = annotation if annotation is not None else doc_type

            rows.append(
                {
                    "parameter": param_name,
                    "type": param_type,
                    "default": default,
                    "description": description,
                }
            )

        return pd.DataFrame(
            rows,
            columns=["parameter", "type", "default", "description"],
        )

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
    def _extract_parameter_docs(method) -> dict:
        """Extract parameter types and descriptions from a NumPy-style docstring."""
        docstring = inspect.getdoc(method)
        if not docstring:
            return {}

        lines = docstring.splitlines()
        params = {}

        # Find "Parameters" section.
        i = 0
        while i < len(lines):
            if lines[i].strip() == "Parameters":
                i += 1
                # Skip dashed separator line(s).
                while i < len(lines) and set(lines[i].strip()) <= {"-"}:
                    i += 1
                break
            i += 1
        else:
            return {}

        # Parse entries until next section header.
        while i < len(lines):
            line = lines[i]

            # Stop at a likely next section header.
            if (
                line.strip()
                and not line.startswith(" ")
                and ":" not in line
                and i + 1 < len(lines)
                and set(lines[i + 1].strip()) <= {"-"}
                and lines[i + 1].strip()
            ):
                break

            # Parameter line pattern: "name : type"
            if line.strip() and not line.startswith(" ") and ":" in line:
                name_part, type_part = line.split(":", 1)
                param_name = name_part.strip()
                param_type = type_part.strip()

                i += 1
                desc_lines = []
                while i < len(lines):
                    next_line = lines[i]
                    if next_line.startswith("    ") or next_line.startswith("\t"):
                        desc_lines.append(next_line.strip())
                        i += 1
                    elif not next_line.strip():
                        # preserve paragraph spacing lightly
                        desc_lines.append("")
                        i += 1
                    else:
                        break

                description = " ".join(x for x in desc_lines if x).strip()
                params[param_name] = {
                    "type": param_type if param_type else None,
                    "description": description if description else None,
                }
                continue

            i += 1

        return params

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