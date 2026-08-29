from src import preprocessing


def test_normalize_lowercases_and_collapses_whitespace():
    assert preprocessing.normalize("I     am      feeling      stressed.") == \
        "i am feeling stressed."


def test_tokenize_splits_words():
    tokens = preprocessing.tokenize("i am feeling stressed")
    assert tokens == ["i", "am", "feeling", "stressed"]


def test_remove_noise_strips_punctuation_tokens():
    tokens = preprocessing.tokenize(preprocessing.normalize("I REALLY love this movie!!!"))
    cleaned = preprocessing.remove_noise(tokens)
    assert "!" not in cleaned
    assert "!!!" not in cleaned
    assert "love" in cleaned


def test_remove_noise_strips_emoji():
    tokens = preprocessing.tokenize(preprocessing.normalize("I am happy 😊!!!"))
    cleaned = preprocessing.remove_noise(tokens)
    assert not any("😊" in t for t in cleaned)


def test_remove_stopwords_removes_common_words():
    tokens = ["i", "am", "feeling", "stressed"]
    cleaned = preprocessing.remove_stopwords(tokens)
    assert "i" not in cleaned
    assert "am" not in cleaned
    assert "feeling" in cleaned or "stressed" in cleaned


def test_lemmatize_reduces_to_base_form():
    lemmatized = preprocessing.lemmatize(["running", "cats"])
    assert lemmatized == ["running", "cat"] or "cat" in lemmatized


def test_preprocess_text_returns_original_and_processed():
    result = preprocessing.preprocess_text("I REALLY love this movie!!!")
    assert result["original_text"] == "I REALLY love this movie!!!"
    assert "!!!" not in result["processed_text"]
    assert "love" in result["processed_text"]


def test_preprocess_text_handles_empty_string():
    result = preprocessing.preprocess_text("")
    assert result["original_text"] == ""
    assert result["processed_text"] == ""


def test_preprocess_text_handles_repeated_spaces():
    result = preprocessing.preprocess_text("I     am      feeling      stressed.")
    assert "  " not in result["processed_text"]


def test_preprocess_text_preserves_original_for_short_and_long_text():
    short = preprocessing.preprocess_text("Good.")
    long_text = "I am extremely unhappy with my workload and feel stressed " * 5
    long_result = preprocessing.preprocess_text(long_text)
    assert short["original_text"] == "Good."
    assert long_result["original_text"] == long_text.strip()
