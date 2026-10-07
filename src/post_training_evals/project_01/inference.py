from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .data import load_yaml
from .grading import grade_response
from .models import LoadedVLM


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_model(
    model_name: str,
    config_path: str | Path = "configs/project_01.yaml",
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    if model_name not in config["models"]:
        raise ValueError(f"Unknown model alias: {model_name}")
    model_config = config["models"][model_name]
    model_id = str(model_config["id"])
    model_revision = str(model_config["revision"])
    cases = _read_jsonl(Path(config["data"]["cases"]))
    if limit is not None:
        cases = cases[:limit]

    output_dir = Path(config["outputs"]["runs"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_predictions.jsonl"
    completed: set[str] = set()
    if output_path.exists():
        for row in _read_jsonl(output_path):
            if not row.get("error"):
                completed.add(str(row["case_id"]))

    runner = LoadedVLM.load(
        model_id,
        model_revision,
        str(config["inference"]["dtype"]),
    )
    torch = runner.torch
    torch.manual_seed(int(config["seed"]))
    processed = 0
    failures = 0
    with output_path.open("a", encoding="utf-8") as output:
        for case in cases:
            if case["case_id"] in completed:
                continue
            started = time.perf_counter()
            result: dict[str, Any] = {
                **case,
                "model_alias": model_name,
                "model_id": model_id,
                "model_revision": runner.revision,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "generation": {
                    "do_sample": False,
                    "max_new_tokens": int(config["inference"]["max_new_tokens"]),
                    "dtype": str(config["inference"]["dtype"]),
                    "device": "cuda",
                },
            }
            try:
                with Image.open(case["image_path"]) as opened:
                    image = opened.convert("RGB")
                raw_text, stats = runner.generate(
                    image,
                    case["prompt"],
                    max_new_tokens=int(config["inference"]["max_new_tokens"]),
                )
                torch.cuda.synchronize()
                result.update(stats)
                result["raw_text"] = raw_text
                result.update(
                    grade_response(raw_text, case["expected_class"], case["grading_mode"])
                )
                result["error"] = None
            except Exception as exc:  # per-case durability is intentional
                failures += 1
                result.update(
                    {
                        "raw_text": "",
                        "normalized_label": None,
                        "is_correct": False,
                        "format_valid": False,
                        "response_chars": 0,
                        "response_words": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            result["latency_seconds"] = time.perf_counter() - started
            output.write(json.dumps(result, sort_keys=True) + "\n")
            output.flush()
            processed += 1
            if processed == 1 or processed % 25 == 0:
                print(
                    f"[{model_name}] completed {processed} new cases "
                    f"({len(completed) + processed}/{len(cases)} total); failures={failures}",
                    flush=True,
                )

    return {
        "model": model_name,
        "model_id": model_id,
        "processed": processed,
        "skipped": len(cases) - processed,
        "failures": failures,
        "output": str(output_path),
    }
