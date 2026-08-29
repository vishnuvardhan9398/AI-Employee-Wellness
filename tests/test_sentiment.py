from src import sentiment


def test_analyze_sentiment_returns_expected_fields():
    result = sentiment.analyze_sentiment("The team meeting is scheduled for Monday.")
    for key in ("positive", "negative", "neutral", "compound", "classification"):
        assert key in result


def test_analyze_sentiment_scores_are_numeric():
    result = sentiment.analyze_sentiment("I really enjoy my work and feel happy with my team.")
    assert isinstance(result["positive"], float)
    assert isinstance(result["negative"], float)
    assert isinstance(result["neutral"], float)
    assert isinstance(result["compound"], float)


def test_positive_text_classified_positive():
    result = sentiment.analyze_sentiment(
        "I really enjoy my work and feel happy with my team."
    )
    assert result["classification"] == "Positive"
    assert result["compound"] >= sentiment.POSITIVE_THRESHOLD


def test_negative_text_classified_negative():
    result = sentiment.analyze_sentiment(
        "I am extremely unhappy with my workload and feel stressed."
    )
    assert result["classification"] == "Negative"
    assert result["compound"] <= sentiment.NEGATIVE_THRESHOLD


def test_neutral_text_classified_neutral():
    result = sentiment.analyze_sentiment("The team meeting is scheduled for Monday.")
    assert result["classification"] == "Neutral"


def test_classification_is_dynamic_not_hardcoded():
    # Two different positive-leaning texts should be able to produce
    # different compound scores — proving the value isn't a fixed constant.
    r1 = sentiment.analyze_sentiment("I like my job.")
    r2 = sentiment.analyze_sentiment("I absolutely love my job, it's amazing and wonderful!")
    assert r1["compound"] != r2["compound"]


def test_classify_compound_boundaries():
    assert sentiment.classify_compound(0.05) == "Positive"
    assert sentiment.classify_compound(-0.05) == "Negative"
    assert sentiment.classify_compound(0.0) == "Neutral"


def test_multiple_additional_examples_produce_appropriate_outputs():
    samples = {
        "My manager appreciated my effort on the project.": "Positive",
        "I feel overwhelmed and exhausted by the constant deadlines.": "Negative",
        "The office is located on the third floor.": "Neutral",
        "I love collaborating with such a supportive team!": "Positive",
        "This has been a terrible, frustrating week at work.": "Negative",
    }
    for text, expected in samples.items():
        result = sentiment.analyze_sentiment(text)
        assert result["classification"] == expected, f"Unexpected result for: {text}"
