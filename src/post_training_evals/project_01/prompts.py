from __future__ import annotations

from dataclasses import dataclass


CLASSES = (
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
)


@dataclass(frozen=True, slots=True)
class PromptSpec:
    prompt_id: str
    text: str
    grading_mode: str


def prompt_specs() -> tuple[PromptSpec, ...]:
    labels = ", ".join(CLASSES)
    return (
        PromptSpec("open", "What is the primary subject of this image?", "semantic"),
        PromptSpec(
            "closed_set",
            f"Classify the primary subject of this image as exactly one of: {labels}.",
            "semantic",
        ),
        PromptSpec(
            "strict_label",
            f"Return only one lowercase class label from this list and no other text: {labels}.",
            "strict_label",
        ),
        PromptSpec(
            "json_label",
            'Return exactly {"label":"<class>"} with no additional keys or text, '
            f"where <class> is one of: {labels}.",
            "json_label",
        ),
    )
