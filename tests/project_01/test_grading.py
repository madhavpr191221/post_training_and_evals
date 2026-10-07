import pytest

from post_training_evals.project_01.grading import grade_response, normalized_label


@pytest.mark.parametrize(
    ("response", "expected"),
    [("A macaque.", "monkey"), ("aircraft", "airplane"), ("The subject is a cat.", "cat")],
)
def test_semantic_aliases(response: str, expected: str) -> None:
    assert normalized_label(response) == expected


def test_ambiguous_response_is_not_forced_to_one_label() -> None:
    assert normalized_label("a cat and a dog") is None


def test_strict_label_separates_correctness_from_format() -> None:
    grade = grade_response("The image contains a cat.", "cat", "strict_label")
    assert grade["is_correct"] is True
    assert grade["format_valid"] is False


def test_json_requires_exact_schema() -> None:
    valid = grade_response('{"label":"ship"}', "ship", "json_label")
    invalid = grade_response('{"label":"ship","confidence":1}', "ship", "json_label")
    assert valid["is_correct"] is True and valid["format_valid"] is True
    assert invalid["is_correct"] is True and invalid["format_valid"] is False
