from __future__ import annotations

import json
from pathlib import Path

import yaml

from post_training_evals.project_01.prompts import prompt_specs
from post_training_evals.project_01.reporting import build_report


def test_report_explains_prompt_suite_metrics_and_pairing(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    rows = {
        "base": [
            {"case_id": "cat-001:open", "expected_class": "cat", "prompt_id": "open", "grading_mode": "semantic", "is_correct": True, "raw_text": "cat", "response_words": 1, "latency_seconds": 1.0, "peak_vram_bytes": 1024, "error": None},
            {"case_id": "dog-001:open", "expected_class": "dog", "prompt_id": "open", "grading_mode": "semantic", "is_correct": False, "raw_text": "", "response_words": 0, "latency_seconds": 1.0, "peak_vram_bytes": 1024, "error": None},
        ],
        "instruct": [
            {"case_id": "cat-001:open", "expected_class": "cat", "prompt_id": "open", "grading_mode": "semantic", "is_correct": False, "raw_text": "dog", "response_words": 1, "latency_seconds": 1.0, "peak_vram_bytes": 1024, "error": None},
            {"case_id": "dog-001:open", "expected_class": "dog", "prompt_id": "open", "grading_mode": "semantic", "is_correct": True, "raw_text": "dog", "response_words": 1, "latency_seconds": 1.0, "peak_vram_bytes": 1024, "error": None},
        ],
    }
    for alias, model_rows in rows.items():
        (run_dir / f"{alias}_predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in model_rows), encoding="utf-8"
        )

    report_path = tmp_path / "report.md"
    config = {
        "outputs": {"runs": str(run_dir), "report": str(report_path)},
        "models": {
            alias: {"id": f"example/{alias}", "revision": "abc123"}
            for alias in ("base", "instruct")
        },
        "inference": {"device": "cuda", "dtype": "bfloat16", "do_sample": False, "max_new_tokens": 48},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    build_report(config_path)
    report = report_path.read_text(encoding="utf-8")

    assert all(spec.text in report for spec in prompt_specs())
    assert "Semantic accuracy" in report
    assert "JSON accuracy is therefore not JSON validity" in report
    assert "latency" in report and "excludes writing the result row to disk" in report
    assert "| Both Correct | 0 | 0.0% |" in report
    assert "| Base Only | 1 | 50.0% |" in report
    assert "| Instruct Only | 1 | 50.0% |" in report
    assert "96x96" not in report
