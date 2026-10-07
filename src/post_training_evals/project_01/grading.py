from __future__ import annotations

import json
import re
from typing import Any

from .prompts import CLASSES


ALIASES = {
    "airplane": {"airplane", "aeroplane", "aircraft", "plane", "jet"},
    "bird": {"bird", "avian"},
    "car": {"car", "automobile", "auto", "sedan"},
    "cat": {"cat", "kitten", "feline"},
    "deer": {"deer", "doe", "buck", "stag"},
    "dog": {"dog", "puppy", "canine"},
    "horse": {"horse", "pony", "mare", "stallion", "equine"},
    "monkey": {"monkey", "macaque", "primate"},
    "ship": {"ship", "boat", "vessel"},
    "truck": {"truck", "lorry"},
}


def normalized_label(text: str) -> str | None:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    exact_matches = [label for label, aliases in ALIASES.items() if cleaned in aliases]
    if len(exact_matches) == 1:
        return exact_matches[0]
    found: set[str] = set()
    for label, aliases in ALIASES.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", cleaned) for alias in aliases):
            found.add(label)
    return next(iter(found)) if len(found) == 1 else None


def grade_response(raw_text: str, expected_class: str, grading_mode: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    format_valid = True
    parsed_label: str | None

    if grading_mode == "json_label":
        try:
            parsed = json.loads(raw_text)
            format_valid = (
                isinstance(parsed, dict)
                and set(parsed) == {"label"}
                and isinstance(parsed["label"], str)
                and parsed["label"] in CLASSES
            )
            parsed_label = normalized_label(parsed.get("label", "")) if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            format_valid = False
            parsed_label = normalized_label(raw_text)
    elif grading_mode == "strict_label":
        format_valid = raw_text in CLASSES
        parsed_label = normalized_label(raw_text)
    elif grading_mode == "semantic":
        parsed_label = normalized_label(raw_text)
    else:
        raise ValueError(f"Unknown grading mode: {grading_mode}")

    return {
        "normalized_label": parsed_label,
        "is_correct": parsed_label == expected_class,
        "format_valid": format_valid,
        "response_chars": len(raw_text),
        "response_words": len(raw_text.split()),
    }
