"""Tests for the safe_query module (GH-1292).

Ensures that the AST-based query validator correctly accepts safe expressions
and rejects malicious ones that could lead to arbitrary code execution via
pandas DataFrame.eval() / DataFrame.query().
"""

import pytest
from typeguard import TypeCheckError

from malariagen_data.anoph.safe_query import UnsafeQueryError, validate_query


class TestValidateQueryAcceptsSafe:
    """Ensure legitimate pandas query expressions are accepted."""

    def test_simple_equality(self):
        validate_query("country == 'Ghana'")

    def test_numeric_comparison(self):
        validate_query("size >= 10")

    def test_boolean_and(self):
        validate_query("country == 'Ghana' and year == 2020")

    def test_boolean_or(self):
        validate_query("country == 'Ghana' or country == 'Mali'")

    def test_not_operator(self):
        validate_query("not country == 'Ghana'")

    def test_parenthesized_expression(self):
        validate_query("(country == 'Ghana') and (year > 2015)")

    def test_in_operator_with_tuple(self):
        validate_query("country in ('Ghana', 'Mali', 'Kenya')")

    def test_not_in_operator(self):
        validate_query("country not in ('Ghana', 'Mali')")

    def test_complex_boolean_chain(self):
        validate_query("country == 'Ghana' and year >= 2015 and taxon == 'gambiae'")

    def test_numeric_arithmetic(self):
        validate_query("size + 1 > 10")

    def test_is_comparison(self):
        validate_query("value is None")

    def test_is_not_comparison(self):
        validate_query("value is not None")

    def test_boolean_literal_true(self):
        validate_query("is_surveillance == True")

    def test_boolean_literal_false(self):
        validate_query("is_surveillance == False")

    def test_with_allowed_names(self):
        validate_query(
            "country == 'Ghana'",
            allowed_names={"country", "year", "taxon"},
        )

    def test_inequality_operators(self):
        validate_query("year != 2020")
        validate_query("size < 100")
        validate_query("size <= 100")
        validate_query("size > 0")
        validate_query("size >= 1")

    def test_unary_minus(self):
        validate_query("value > -1")

    def test_list_literal_in(self):
        validate_query("country in ['Ghana', 'Mali']")

    def test_whitespace_handling(self):
        validate_query("  country == 'Ghana'  ")

    def test_at_variable_reference(self):
        """Pandas @var syntax for referencing local variables."""
        validate_query("sex_call in @sex_call_list")

    def test_at_variable_in_compound(self):
        validate_query("taxon in @taxon_list and year > 2015")

    def test_at_variable_equality(self):
        validate_query("country == @target_country")


class TestValidateQueryRejectsMalicious:
    """Ensure that code injection attempts are blocked."""

    def test_import_call(self):
        with pytest.raises(UnsafeQueryError):
            validate_query("__import__('os').system('echo PWNED')")

    def test_import_in_compound_expression(self):
        with pytest.raises(UnsafeQueryError):
            validate_query(
                "__import__('os').system('echo PWNED') or country == 'Ghana'"
            )

    def test_function_call(self):
        with pytest.raises(UnsafeQueryError):
            validate_query("len(country)")

    def test_attribute_access(self):
        with pytest.raises(UnsafeQueryError):
            validate_query("country.upper()")

    def test_nested_attribute_access(self):
        with pytest.raises(UnsafeQueryError):
            validate_query("os.system('id')")

    def test_subscript(self):
        with pytest.raises(UnsafeQueryError):
            validate_query("country[0]")

    def test_lambda(self):
        with pytest.raises(UnsafeQueryError):
            validate_query("lambda: 1")

    def test_list_comprehension(self):
        with pytest.raises(UnsafeQueryError):
            validate_query("[x for x in range(10)]")

    def test_dunder_identifier(self):
        with pytest.raises(UnsafeQueryError):
            validate_query("__class__")

    def test_dunder_in_name(self):
        with pytest.raises(UnsafeQueryError):
            validate_query("__builtins__.__import__")

    def test_exec_call(self):
        with pytest.raises(UnsafeQueryError):
            validate_query("exec('import os')")

    def test_eval_call(self):
        with pytest.raises(UnsafeQueryError):
            validate_query("eval('1+1')")

    def test_dict_literal(self):
        with pytest.raises(UnsafeQueryError):
            validate_query("{'key': 'value'}")

    def test_generator_expression(self):
        with pytest.raises(UnsafeQueryError):
            validate_query("sum(x for x in range(10))")

    def test_starred_expression(self):
        with pytest.raises(UnsafeQueryError):
            validate_query("*args")

    def test_fstring_attempt(self):
        with pytest.raises(UnsafeQueryError):
            validate_query("f'{__import__(\"os\")}'")

    def test_walrus_operator(self):
        with pytest.raises(UnsafeQueryError):
            validate_query("(x := 1)")

    def test_unknown_column_with_allowlist(self):
        with pytest.raises(UnsafeQueryError, match="Unknown column name"):
            validate_query(
                "evil_col == 'value'",
                allowed_names={"country", "year"},
            )


class TestValidateQueryEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_string(self):
        with pytest.raises(UnsafeQueryError, match="must not be empty"):
            validate_query("")

    def test_whitespace_only(self):
        with pytest.raises(UnsafeQueryError, match="must not be empty"):
            validate_query("   ")

    def test_non_string_input(self):
        with pytest.raises((UnsafeQueryError, TypeError, TypeCheckError)):
            validate_query(123)

    def test_syntax_error(self):
        with pytest.raises(UnsafeQueryError, match="not a valid expression"):
            validate_query("country ==")

    def test_multiple_statements(self):
        # Multiple statements can't be parsed in eval mode
        with pytest.raises(UnsafeQueryError):
            validate_query("x = 1; y = 2")

    def test_quote_breaking_attempt(self):
        """Ensure quote-breaking in string literals doesn't bypass validation."""
        with pytest.raises(UnsafeQueryError):
            validate_query("contig == 'X' or __import__('os').system('id')")
