from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .data import load_yaml
from .prompts import CLASSES, prompt_specs


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    latest = {str(row["case_id"]): row for row in rows}
    return list(latest.values())


def _rate(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [bool(row.get(key)) for row in rows]
    return sum(values) / len(values) if values else 0.0


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if not row.get("error")]
    format_checked = [
        row
        for row in successful
        if row.get("grading_mode") in {"strict_label", "json_label"}
    ]
    latencies = [float(row["latency_seconds"]) for row in successful]
    return {
        "cases": len(rows),
        "successful": len(successful),
        "accuracy": _rate(successful, "is_correct"),
        "format_valid_rate": _rate(format_checked, "format_valid") if format_checked else None,
        "mean_response_words": statistics.fmean(
            float(row["response_words"]) for row in successful
        ) if successful else 0.0,
        "median_latency_seconds": statistics.median(latencies) if latencies else 0.0,
        "peak_vram_mib": max(
            (int(row.get("peak_vram_bytes", 0)) / (1024**2) for row in successful),
            default=0.0,
        ),
    }


def build_report(config_path: str | Path = "configs/project_01.yaml") -> dict[str, Any]:
    config = load_yaml(config_path)
    run_dir = Path(config["outputs"]["runs"])
    model_rows = {
        alias: _read_jsonl(run_dir / f"{alias}_predictions.jsonl")
        for alias in ("base", "instruct")
    }
    by_model_case = {
        alias: {str(row["case_id"]): row for row in rows}
        for alias, rows in model_rows.items()
    }
    shared_cases = sorted(set(by_model_case["base"]) & set(by_model_case["instruct"]))

    comparison: dict[str, Any] = {
        "models": {
            alias: {
                **_summary(rows),
                "model_id": str(config["models"][alias]["id"]),
                "model_revision": str(config["models"][alias]["revision"]),
            }
            for alias, rows in model_rows.items()
        },
        "shared_cases": len(shared_cases),
        "paired": {"both_correct": 0, "base_only": 0, "instruct_only": 0, "both_wrong": 0},
        "by_class": {},
        "by_prompt": {},
    }
    for case_id in shared_cases:
        base_correct = bool(by_model_case["base"][case_id].get("is_correct"))
        instruct_correct = bool(by_model_case["instruct"][case_id].get("is_correct"))
        key = (
            "both_correct" if base_correct and instruct_correct else
            "base_only" if base_correct else
            "instruct_only" if instruct_correct else
            "both_wrong"
        )
        comparison["paired"][key] += 1

    for grouping_key, output_key, values in (
        ("expected_class", "by_class", CLASSES),
        ("prompt_id", "by_prompt", ("open", "closed_set", "strict_label", "json_label")),
    ):
        for value in values:
            comparison[output_key][value] = {
                alias: _summary([row for row in rows if row[grouping_key] == value])
                for alias, rows in model_rows.items()
            }

    disagreements = []
    for case_id in shared_cases:
        base = by_model_case["base"][case_id]
        instruct = by_model_case["instruct"][case_id]
        if base.get("is_correct") != instruct.get("is_correct"):
            disagreements.append(
                {
                    "case_id": case_id,
                    "expected": base["expected_class"],
                    "base": base.get("raw_text", ""),
                    "instruct": instruct.get("raw_text", ""),
                }
            )
    disagreements.sort(
        key=lambda row: (
            row["case_id"].split(":", 1)[1],
            row["case_id"].split(":", 1)[0].split("-", 1)[0],
        )
    )
    chosen: list[dict[str, Any]] = []
    per_group: defaultdict[tuple[str, str], int] = defaultdict(int)
    for row in disagreements:
        image_id, prompt_id = row["case_id"].split(":", 1)
        class_name = image_id.split("-", 1)[0]
        group = (prompt_id, class_name)
        if per_group[group] >= 2:
            continue
        chosen.append(row)
        per_group[group] += 1
        if len(chosen) == 20:
            break
    comparison["representative_disagreements"] = chosen

    run_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = run_dir / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")

    report_path = Path(config["outputs"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Project 1: SmolVLM Base vs Instruct",
        "",
        "This is an inference-only paired comparison: the two checkpoints were evaluated on identical image-prompt cases; no training or parameter updates occurred.",
        "",
        "## Evaluation setup",
        "",
        f"- Base: `{comparison['models']['base']['model_id']}` at revision "
        f"`{comparison['models']['base']['model_revision']}`",
        f"- Instruct: `{comparison['models']['instruct']['model_id']}` at revision "
        f"`{comparison['models']['instruct']['model_revision']}`",
        "- Dataset: 300 visually reviewed web-search images, balanced at 30 images in each of ten folder-label classes: "
        + ", ".join(CLASSES) + ".",
        "- Selected source files were copied without resizing or re-encoding; image dimensions therefore vary. Folder labels were based on search-query curation and visual review, not independent annotation.",
        "- Each image was evaluated with all four prompts, producing 1,200 image-prompt cases per model. A case means one image paired with one prompt; the four cases from the same image are related, not four independent images.",
        f"- Generation used `{config['inference']['device']}` with `{config['inference']['dtype']}`, deterministic decoding (`do_sample={str(config['inference']['do_sample']).lower()}`), and a maximum of {config['inference']['max_new_tokens']} new tokens.",
        "- The processor formatted each request as a user message containing the image and prompt. Model responses were decoded and stripped of surrounding whitespace before grading.",
        "",
        "## Prompts",
        "",
        "The prompt suite varies both how constrained the answer is and whether a machine-readable format is requested:",
        "",
        "| Condition | Exact prompt | Purpose |",
        "|---|---|---|",
    ]
    purposes = {
        "open": "Open-ended identification, without a supplied label list.",
        "closed_set": "Choose a class from the ten listed labels, while allowing a natural-language response.",
        "strict_label": "Test exact single-label output: one lowercase class token and no other text.",
        "json_label": "Test exact structured output with one `label` key and no extra text or keys.",
    }
    for spec in prompt_specs():
        lines.append(f"| `{spec.prompt_id}` | {spec.text} | {purposes[spec.prompt_id]} |")
    lines.extend([
        "",
        "## What was measured and how responses were scored",
        "",
        "- **Semantic accuracy** is the fraction of successful cases whose response could be mapped to exactly one expected class; failed inference cases are excluded from this denominator. The grader lowercases and normalizes text, then recognizes the class names and configured aliases (for example, `aircraft` for airplane and `puppy` for dog). A response mentioning multiple recognized classes, or none, is not mapped to a class. This is a heuristic text grader, not human judgment of visual understanding.",
        "- **Format compliance** is separate from semantic accuracy. For `strict_label`, the stripped response must exactly equal one lowercase class token. For `json_label`, it must parse as a JSON object with exactly one `label` key whose value is exactly one of the ten lowercase class tokens. Open and closed-set prompts have no format target and are excluded from this measure; the overall compliance denominator is therefore the successful strict-label plus JSON cases (600 per model here).",
        "- Consequently, a semantically correct answer can fail format compliance: `Cat.` can be correct for the cat class but fail strict-label formatting, and plain text `cat` can be semantically correct on the JSON prompt while failing JSON formatting. For malformed JSON, the grader still attempts to find a semantic class in the raw response; JSON accuracy is therefore not JSON validity.",
        "- **Overall accuracy** pools the four prompt conditions (all image-prompt cases); it is not an estimate from one prompt or a prompt-independent measure of visual recognition. Prompt-level accuracy is shown separately below.",
        "- **Mean words** is the arithmetic mean of whitespace-separated words in decoded responses. **Median latency** is the median per-case wall time, including image opening/conversion, processor preparation, generation, CUDA synchronization, decoding, and grading; it is not pure model decoding time and excludes writing the result row to disk. **Peak VRAM** is the largest per-generation PyTorch CUDA memory allocation reported across cases, not total device memory usage.",
        "",
        "## Overall results",
        "",
        "| Model | Cases | Successful | Semantic accuracy | Strict/JSON format compliance | Mean words | Median latency (s) | Peak allocated VRAM (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for alias in ("base", "instruct"):
        summary = comparison["models"][alias]
        format_rate = summary["format_valid_rate"]
        format_text = f"{format_rate:.1%}" if format_rate is not None else "n/a"
        lines.append(
            f"| {alias} | {summary['cases']} | {summary['successful']} | {summary['accuracy']:.1%} | "
            f"{format_text} | {summary['mean_response_words']:.2f} | "
            f"{summary['median_latency_seconds']:.3f} | {summary['peak_vram_mib']:.1f} |"
        )
    lines.extend([
        "",
        "## Paired outcomes",
        "",
        "For each shared image-prompt case, compare whether each model was semantically correct. The categories below are mutually exclusive and partition the shared cases:",
        "",
        "| Outcome | Cases | Share | Meaning |",
        "|---|---:|---:|---|",
    ])
    outcome_descriptions = {
        "both_correct": "Both models were judged semantically correct.",
        "base_only": "Base was correct; Instruct was not.",
        "instruct_only": "Instruct was correct; Base was not.",
        "both_wrong": "Neither model was judged correct.",
    }
    for key, value in comparison["paired"].items():
        share = value / len(shared_cases) if shared_cases else 0.0
        lines.append(f"| {key.replace('_', ' ').title()} | {value} | {share:.1%} | {outcome_descriptions[key]} |")
    base_correct = comparison["paired"]["both_correct"] + comparison["paired"]["base_only"]
    instruct_correct = comparison["paired"]["both_correct"] + comparison["paired"]["instruct_only"]
    lines.extend([
        "",
        f"These counts imply {base_correct} semantically correct Base cases and {instruct_correct} semantically correct Instruct cases among {len(shared_cases)} paired cases. In the {comparison['paired']['base_only'] + comparison['paired']['instruct_only']} cases where correctness differed, Instruct was correct on {comparison['paired']['instruct_only']} and Base on {comparison['paired']['base_only']}. The outcome shares can differ from 100% by a tenth of a percentage point because of rounding.",
        "",
        "The paired cases are not independent image samples: each image contributes four prompt conditions. These counts describe this image set and prompt suite; they do not identify the cause of the model difference or establish performance on other images or prompts.",
    ])
    lines.extend(["", "## Accuracy and format compliance by prompt", ""])
    lines.extend(["| Prompt | Base accuracy | Instruct accuracy | Base format | Instruct format |", "|---|---:|---:|---:|---:|"])
    for prompt_id, values in comparison["by_prompt"].items():
        base_format = values["base"]["format_valid_rate"]
        instruct_format = values["instruct"]["format_valid_rate"]
        base_format_text = f"{base_format:.1%}" if base_format is not None else "n/a"
        instruct_format_text = (
            f"{instruct_format:.1%}" if instruct_format is not None else "n/a"
        )
        lines.append(
            f"| {prompt_id} | {values['base']['accuracy']:.1%} | "
            f"{values['instruct']['accuracy']:.1%} | "
            f"{base_format_text} | {instruct_format_text} |"
        )
    lines.extend(["", "## Accuracy by class", "", "Each class row pools its 30 images across four prompts (120 cases per model)."])
    lines.extend(["| Class | Base | Instruct |", "|---|---:|---:|"])
    for class_name, values in comparison["by_class"].items():
        lines.append(
            f"| {class_name} | {values['base']['accuracy']:.1%} | "
            f"{values['instruct']['accuracy']:.1%} |"
        )
    lines.extend(["", "## Model behavior notes", ""])
    lines.append(
        "Instruct usually returned a class label, but its strict-label outputs often included capitalization or punctuation, and its JSON-prompt outputs did not parse as valid JSON. Base frequently emitted empty or image-grid placeholder text under constrained prompts. These are observed response patterns; semantic correctness and requested-format compliance remain distinct outcomes."
    )
    lines.extend(["", "## Interpretation boundary", ""])
    lines.append(
        "These descriptive results apply to this audited image set and prompt suite. "
        "The source labels are coarse object classes and may not capture every reasonable interpretation of an image; review was not independent annotation. The corpus is a curated web-search sample rather than a probability sample. "
        "These results do not establish a causal or population-level effect of instruction tuning."
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"comparison": str(comparison_path), "report": str(report_path), **comparison}
