from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import yaml

from .prompts import CLASSES, prompt_specs


@dataclass(frozen=True, slots=True)
class SourceRecord:
    class_name: str
    saved_name: str
    source_url: str
    source_query: str
    source_file: str
    source_path: Path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def _stable_order_key(seed: int, record: SourceRecord) -> str:
    raw = f"{seed}:{record.class_name}:{record.saved_name}".encode()
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source_records(source_root: Path, source_manifest: Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    with source_manifest.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            class_name = str(row["class_name"]).strip()
            saved_name = str(row["saved_name"]).strip()
            if class_name not in CLASSES:
                continue
            source_path = source_root / class_name / saved_name
            if not source_path.is_file():
                continue
            records.append(
                SourceRecord(
                    class_name=class_name,
                    saved_name=saved_name,
                    source_url=str(row.get("source_url", "")).strip(),
                    source_query=str(row.get("source_query", "")).strip(),
                    source_file=str(row.get("source_file", "")).strip(),
                    source_path=source_path,
                )
            )
    return records


def _rejections(path: Path) -> dict[str, set[str]]:
    value = load_yaml(path).get("reject", {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("audit reject must map class names to filename lists")
    return {str(key): {str(item) for item in items or []} for key, items in value.items()}


def _make_contact_sheet(records: list[SourceRecord], output_path: Path) -> None:
    columns, tile, label_height = 5, 128, 24
    rows = math.ceil(len(records) / columns)
    sheet = Image.new("RGB", (columns * tile, rows * (tile + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, record in enumerate(records):
        row, column = divmod(index, columns)
        x, y = column * tile, row * (tile + label_height)
        with Image.open(record.source_path) as source:
            image = source.convert("RGB")
            image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
            offset_x = (tile - image.width) // 2
            offset_y = (tile - image.height) // 2
            sheet.paste(image, (x + offset_x, y + offset_y))
        draw.text((x + 3, y + tile + 4), record.saved_name, fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_data(
    config_path: str | Path = "configs/project_01.yaml",
    audit_config_path: str | Path = "configs/project_01_audit.yaml",
) -> dict[str, int]:
    config = load_yaml(config_path)
    seed = int(config["seed"])
    per_class = int(config["per_class"])
    candidate_pool_size = int(config["candidate_pool_per_class"])
    source_root = Path(config["source"]["root"])
    source_manifest = Path(config["source"]["manifest"])
    image_root = Path(config["data"]["images"])
    output_manifest = Path(config["data"]["manifest"])
    audit_path = Path(config["data"]["audit"])
    cases_path = Path(config["data"]["cases"])
    contact_root = Path(config["data"]["contact_sheets"])
    rejected = _rejections(Path(audit_config_path))

    records = _read_source_records(source_root, source_manifest)
    by_class = {class_name: [] for class_name in CLASSES}
    for record in records:
        by_class[record.class_name].append(record)
    for class_name in CLASSES:
        by_class[class_name].sort(key=lambda record: _stable_order_key(seed, record))
        if len(by_class[class_name]) < per_class:
            raise RuntimeError(f"Only {len(by_class[class_name])} eligible {class_name} images")

    for class_name in CLASSES:
        _make_contact_sheet(
            by_class[class_name][:candidate_pool_size],
            contact_root / f"{class_name}_candidates.png",
        )

    selected: list[SourceRecord] = []
    audit_rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for class_name in CLASSES:
        accepted_count = 0
        for rank, record in enumerate(by_class[class_name], start=1):
            digest = sha256_file(record.source_path)
            if record.saved_name in rejected.get(class_name, set()):
                decision, reason = "rejected", "visual audit rejection"
            elif digest in seen_hashes:
                decision, reason = "rejected", "exact duplicate"
            else:
                decision, reason = "accepted", "visually reviewed candidate"
                accepted_count += 1
                selected.append(record)
                seen_hashes.add(digest)
            audit_rows.append(
                {
                    "class_name": class_name,
                    "candidate_rank": rank,
                    "saved_name": record.saved_name,
                    "sha256": digest,
                    "decision": decision,
                    "reason": reason,
                }
            )
            if accepted_count == per_class:
                break
        if accepted_count != per_class:
            raise RuntimeError(f"Could not select {per_class} reviewed images for {class_name}")

    selected_targets: set[Path] = set()
    manifest_rows: list[dict[str, Any]] = []
    for record in selected:
        target = image_root / record.class_name / record.saved_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.source_path, target)
        selected_targets.add(target.resolve())
        digest = sha256_file(target)
        image_id = f"{record.class_name}-{Path(record.saved_name).stem}"
        manifest_rows.append(
            {
                "image_id": image_id,
                "class_name": record.class_name,
                "saved_name": record.saved_name,
                "image_path": target.as_posix(),
                "source_relative_path": f"{record.class_name}/{record.saved_name}",
                "source_url": record.source_url,
                "source_query": record.source_query,
                "source_file": record.source_file,
                "sha256": digest,
                "audit_status": "accepted",
                "redistribution_status": "unverified",
                "selection_seed": seed,
            }
        )

    for existing in image_root.rglob("*.png") if image_root.exists() else ():
        if existing.resolve() not in selected_targets:
            existing.unlink()

    _write_csv(
        output_manifest,
        manifest_rows,
        [
            "image_id", "class_name", "saved_name", "image_path",
            "source_relative_path", "source_url", "source_query", "source_file",
            "sha256", "audit_status", "redistribution_status", "selection_seed",
        ],
    )
    _write_csv(
        audit_path,
        audit_rows,
        ["class_name", "candidate_rank", "saved_name", "sha256", "decision", "reason"],
    )

    cases_path.parent.mkdir(parents=True, exist_ok=True)
    case_count = 0
    with cases_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            for prompt in prompt_specs():
                case = {
                    "case_id": f"{row['image_id']}:{prompt.prompt_id}",
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "image_sha256": row["sha256"],
                    "expected_class": row["class_name"],
                    "prompt_id": prompt.prompt_id,
                    "prompt": prompt.text,
                    "grading_mode": prompt.grading_mode,
                }
                handle.write(json.dumps(case, sort_keys=True) + "\n")
                case_count += 1

    return {"images": len(manifest_rows), "cases": case_count, "audit_rows": len(audit_rows)}
