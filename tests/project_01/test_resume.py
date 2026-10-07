from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import yaml

from post_training_evals.project_01 import inference


class _Cuda:
    @staticmethod
    def synchronize() -> None:
        return None


class _Torch:
    cuda = _Cuda()

    @staticmethod
    def manual_seed(seed: int) -> None:
        return None


class _Runner:
    torch = _Torch()
    revision = "fake-revision"

    def generate(self, image, prompt: str, *, max_new_tokens: int):
        return "cat", {"input_tokens": 4, "output_tokens": 1, "peak_vram_bytes": 100}


def test_completed_cases_are_resumed_without_duplication(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "cat.png"
    Image.new("RGB", (8, 8), "gray").save(image_path)
    case = {
        "case_id": "cat-1:strict_label",
        "image_id": "cat-1",
        "image_path": str(image_path),
        "image_sha256": "unused",
        "expected_class": "cat",
        "prompt_id": "strict_label",
        "prompt": "Return cat",
        "grading_mode": "strict_label",
    }
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    config = {
        "seed": 1,
        "models": {"base": {"id": "fake/base", "revision": "fake-revision"}},
        "data": {"cases": str(cases_path)},
        "inference": {"dtype": "bfloat16", "max_new_tokens": 8},
        "outputs": {"runs": str(tmp_path / "runs")},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(inference.LoadedVLM, "load", lambda *args, **kwargs: _Runner())

    first = inference.run_model("base", config_path)
    second = inference.run_model("base", config_path)
    rows = (tmp_path / "runs/base_predictions.jsonl").read_text().splitlines()

    assert first["processed"] == 1
    assert second["processed"] == 0
    assert len(rows) == 1
