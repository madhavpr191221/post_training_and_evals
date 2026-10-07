from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
import yaml

from post_training_evals.project_01.data import prepare_data
from post_training_evals.project_01.prompts import CLASSES


def test_prepare_data_is_balanced_and_deterministic(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_manifest = tmp_path / "source.csv"
    rows = []
    for class_index, class_name in enumerate(CLASSES):
        for image_index in range(3):
            saved_name = f"{class_name}_{image_index:03d}.png"
            path = source_root / class_name / saved_name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), (class_index * 20, image_index * 30, 10)).save(path)
            rows.append(
                {
                    "class_name": class_name,
                    "saved_name": saved_name,
                    "source_url": f"https://example.test/{saved_name}",
                    "source_query": class_name,
                    "source_file": saved_name,
                }
            )
    with source_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    config = {
        "seed": 7,
        "per_class": 2,
        "candidate_pool_per_class": 3,
        "source": {"root": str(source_root), "manifest": str(source_manifest)},
        "data": {
            "images": str(tmp_path / "data/images"),
            "manifest": str(tmp_path / "data/manifest.csv"),
            "audit": str(tmp_path / "data/audit.csv"),
            "cases": str(tmp_path / "data/cases.jsonl"),
            "contact_sheets": str(tmp_path / "data/contact_sheets"),
        },
    }
    config_path = tmp_path / "config.yaml"
    audit_path = tmp_path / "audit.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    audit_path.write_text("reject: {}\n", encoding="utf-8")

    first = prepare_data(config_path, audit_path)
    first_manifest = (tmp_path / "data/manifest.csv").read_text(encoding="utf-8")
    second = prepare_data(config_path, audit_path)

    assert first == second == {"images": 20, "cases": 80, "audit_rows": 20}
    assert (tmp_path / "data/manifest.csv").read_text(encoding="utf-8") == first_manifest
    cases = [json.loads(line) for line in (tmp_path / "data/cases.jsonl").read_text().splitlines()]
    assert len(cases) == 80
    assert len({case["case_id"] for case in cases}) == 80
