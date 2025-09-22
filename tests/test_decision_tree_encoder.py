from collections import Counter

import pytest

pl = pytest.importorskip("polars", reason="polars is required for feataz tests")
pytest.importorskip("sklearn", reason="scikit-learn is required for DecisionTreeEncoder tests")

from feataz.encoders import DecisionTreeEncoder


def _priors_from_counter(counter: Counter, classes: list) -> list:
    total = sum(counter.values())
    if total == 0:
        return [0.0 for _ in classes]
    return [counter.get(cls, 0) / total for cls in classes]


def test_decision_tree_encoder_binary_string_labels() -> None:
    train = pl.DataFrame(
        {
            "color": ["red", "blue", "red", "green"],
            "outcome": ["yes", "no", "yes", "yes"],
        }
    )
    encoder = DecisionTreeEncoder(
        target="outcome",
        columns=["color"],
        problem="classification",
        drop_original=False,
        random_state=0,
    )
    encoder.fit(train)

    test = pl.DataFrame({"color": ["red", "yellow", None]})
    transformed = encoder.transform(test)

    tree = encoder.trees_["color"]
    classes = list(getattr(tree, "classes_", []))
    encoded_cols = [c for c in transformed.columns if c.startswith("color__dte__")]
    assert len(encoded_cols) == len(classes) == 2
    assert set(encoded_cols) == {f"color__dte__{cls}" for cls in classes}

    counts = Counter(train.get_column("outcome").to_list())
    expected_priors = _priors_from_counter(counts, classes)
    proba_rows = transformed.select([f"color__dte__{cls}" for cls in classes]).rows()

    for idx, expected in enumerate(expected_priors):
        assert proba_rows[1][idx] == pytest.approx(expected)
        assert proba_rows[2][idx] == pytest.approx(expected)


def test_decision_tree_encoder_multiclass_probabilities() -> None:
    train = pl.DataFrame(
        {
            "shape": [
                "circle",
                "square",
                "triangle",
                "circle",
                "triangle",
                "square",
                "circle",
                "square",
                "circle",
            ],
            "category": [
                "zero",
                "one",
                "two",
                "zero",
                "two",
                "one",
                "zero",
                "zero",
                "one",
            ],
        }
    )
    encoder = DecisionTreeEncoder(
        target="category",
        columns=["shape"],
        problem="classification",
        drop_original=False,
        random_state=0,
    )
    encoder.fit(train)

    test = pl.DataFrame({"shape": ["circle", "hexagon", None]})
    transformed = encoder.transform(test)

    tree = encoder.trees_["shape"]
    classes = list(getattr(tree, "classes_", []))
    encoded_cols = [c for c in transformed.columns if c.startswith("shape__dte__")]
    assert len(encoded_cols) == len(classes) == 3
    assert set(encoded_cols) == {f"shape__dte__{cls}" for cls in classes}

    counts = Counter(train.get_column("category").to_list())
    expected_priors = _priors_from_counter(counts, classes)
    proba_rows = transformed.select([f"shape__dte__{cls}" for cls in classes]).rows()

    for idx, expected in enumerate(expected_priors):
        assert proba_rows[1][idx] == pytest.approx(expected)
        assert proba_rows[2][idx] == pytest.approx(expected)
