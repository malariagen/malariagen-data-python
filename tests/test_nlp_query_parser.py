import datetime as dt

from malariagen_data.nlp_query_parser import NLPQueryParser


def test_parse_sample_query_country_and_year_range():
    parser = NLPQueryParser()

    parsed = parser.parse("Find samples from Kenya collected after 2015")

    assert parsed.task == "sample_metadata"
    assert parsed.country == "Kenya"
    assert parsed.year_min == 2015
    assert parsed.year_max == dt.date.today().year
    assert parsed.sample_query == (
        f"country == 'Kenya' & (year >= 2015) & (year <= {dt.date.today().year})"
    )


def test_parse_multi_population_order_is_deterministic():
    parser = NLPQueryParser()

    parsed = parser.parse("Compare SNP frequencies across Uganda, Kenya, and Tanzania")
    api_call = parser.generate_api_call(parsed)

    assert parsed.task == "snp_frequencies"
    assert parsed.populations == ["Uganda", "Kenya", "Tanzania"]
    assert api_call["function"] == "snp_frequencies"
    assert api_call["params"]["cohorts"] == ["Uganda", "Kenya", "Tanzania"]


def test_parse_fst_and_generate_expected_api_call():
    parser = NLPQueryParser()

    parsed = parser.parse(
        "Where are populations most differentiated between Mali and Cameroon?"
    )
    api_call = parser.generate_api_call(parsed)

    assert parsed.task == "fst"
    assert parsed.populations == ["Mali", "Cameroon"]
    assert api_call["function"] == "fst"
    assert api_call["params"]["cohorts"] == ["Mali", "Cameroon"]
    assert api_call["params"]["region"] == "genome"


def test_parse_map_query_sets_plot_function():
    parser = NLPQueryParser()

    parsed = parser.parse("Plot interactive map of samples from Uganda colored by species")
    api_call = parser.generate_api_call(parsed)

    assert parsed.task == "plot_map"
    assert parsed.visualization == "map"
    assert api_call["function"] == "plot_samples_interactive_map"


def test_parse_explicit_year_range():
    parser = NLPQueryParser()

    parsed = parser.parse("Show Uganda samples from 2012 to 2020")

    assert parsed.year_min == 2012
    assert parsed.year_max == 2020
    assert parsed.sample_query == (
        "country == 'Uganda' & (year >= 2012) & (year <= 2020)"
    )


def test_query_bank_task_classification_accuracy():
    parser = NLPQueryParser()

    query_bank = [
        ("Show me samples from Uganda", "sample_metadata"),
        ("Find Kenya samples collected after 2015", "sample_metadata"),
        ("Perform PCA on Anopheles gambiae samples", "pca"),
        ("Compare SNP frequencies across Uganda and Tanzania", "snp_frequencies"),
        (
            "Where are populations most differentiated between Mali and Cameroon?",
            "fst",
        ),
        ("Plot an interactive map of samples in Benin", "plot_map"),
        ("Visualize sample coordinates across West Africa", "plot_map"),
        ("List all samples from Senegal", "sample_metadata"),
        ("Compute fst divergence between Kenya and Uganda", "fst"),
        ("Show SNP allele frequencies in Uganda", "snp_frequencies"),
    ]

    matches = 0
    for query, expected_task in query_bank:
        parsed = parser.parse(query)
        if parsed.task == expected_task:
            matches += 1

    accuracy = matches / len(query_bank)

    # Keep this threshold realistic for rule-based parsing while preventing regression.
    assert accuracy >= 0.9
