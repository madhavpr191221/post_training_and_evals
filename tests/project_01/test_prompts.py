from post_training_evals.project_01.prompts import CLASSES, prompt_specs


def test_prompt_suite_has_four_unique_conditions() -> None:
    prompts = prompt_specs()
    assert len(prompts) == 4
    assert len({prompt.prompt_id for prompt in prompts}) == 4
    assert all(class_name in prompts[1].text for class_name in CLASSES)
