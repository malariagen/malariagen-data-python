"""Safe query validation for pandas eval/query expressions.

This module provides AST-based validation of query strings to prevent
arbitrary code execution via pandas DataFrame.eval() and DataFrame.query().

Only a restricted subset of Python expressions is allowed:
- Boolean operators: and, or, not
- Comparison operators: ==, !=, <, <=, >, >=, in, not in, is
- Arithmetic operators: +, -, *, /, //, %, **
- Unary operators: +, -, ~, not
- Constants: strings, numbers, booleans, None
- Names: must match an allowlist of known column names (if provided)
- Parenthesized expressions

Forbidden constructs include:
- Function calls (e.g., __import__('os'))
- Attribute access (e.g., os.system)
- Subscript/indexing (e.g., x[0])
- Comprehensions, lambdas, f-strings, starred expressions
- Any identifier containing double underscores (__)
"""

import ast
import re
from typing import Optional, Set

# Pattern matching pandas @variable references in query strings.
# These are not valid Python but are a pandas feature for referencing
# local/global variables via the `local_dict` or `global_dict` kwargs.
_AT_VAR_PATTERN = re.compile(r"@([A-Za-z_][A-Za-z0-9_]*)")


# AST node types that are safe in query expressions.
_SAFE_NODE_TYPES = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Invert,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Tuple,
    ast.List,
)


class UnsafeQueryError(ValueError):
    """Raised when a query string contains unsafe constructs."""

    pass


def _validate_node(node: ast.AST, allowed_names: Optional[Set[str]] = None) -> None:
    """Recursively validate that an AST node contains only safe constructs.

    Parameters
    ----------
    node : ast.AST
        The AST node to validate.
    allowed_names : set of str, optional
        If provided, restrict identifier names to this set.

    Raises
    ------
    UnsafeQueryError
        If the node or any of its children contain unsafe constructs.
    """
    if not isinstance(node, _SAFE_NODE_TYPES):
        raise UnsafeQueryError(
            f"Unsafe expression: {type(node).__name__} nodes are not allowed "
            f"in query strings. Only comparisons, boolean logic, and constants "
            f"are permitted."
        )

    if isinstance(node, ast.Name):
        name = node.id
        # Block dunder identifiers.
        if "__" in name:
            raise UnsafeQueryError(
                f"Unsafe expression: identifier '{name}' contains double "
                f"underscores and is not allowed in query strings."
            )
        # Check against allowlist if provided.
        if allowed_names is not None and name not in allowed_names:
            # Allow common boolean literals that pandas recognizes.
            if name not in {"True", "False", "None"}:
                raise UnsafeQueryError(
                    f"Unknown column name '{name}' in query string. "
                    f"Allowed column names: {sorted(allowed_names)}"
                )

    # Recurse into child nodes.
    for child in ast.iter_child_nodes(node):
        _validate_node(child, allowed_names)


def validate_query(query: str, allowed_names: Optional[Set[str]] = None) -> None:
    """Validate that a query string is safe for use with pandas eval/query.

    Parameters
    ----------
    query : str
        The query string to validate.
    allowed_names : set of str, optional
        If provided, restrict identifier names to this set of known column
        names. If None, any identifier (except those containing ``__``) is
        allowed.

    Raises
    ------
    UnsafeQueryError
        If the query contains unsafe constructs such as function calls,
        attribute access, or dunder identifiers.
    """
    if not isinstance(query, str):
        raise UnsafeQueryError(f"Query must be a string, got {type(query).__name__}.")

    query = query.strip()
    if not query:
        raise UnsafeQueryError("Query string must not be empty.")

    # Replace pandas @variable references with plain identifiers so the
    # expression can be parsed as valid Python.  The replaced names are
    # prefixed with ``_at_`` to avoid collisions with real column names
    # while remaining dunder-free.
    query_for_parse = _AT_VAR_PATTERN.sub(r"_at_\1", query)

    try:
        tree = ast.parse(query_for_parse, mode="eval")
    except SyntaxError as e:
        raise UnsafeQueryError(f"Query string is not a valid expression: {e}") from e

    _validate_node(tree, allowed_names)
