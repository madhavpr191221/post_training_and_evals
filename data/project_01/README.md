# Project 1 image corpus

This directory contains the metadata for a balanced, inference-only visual evaluation.

The source corpus is expected at `../perception_world_models/data/test_images`, with
provenance from its adjacent `test_images_manifest.csv`. Only manifest-backed files are
eligible. Candidates are ordered deterministically from seed `20261007`, visually
audited, and copied without resizing or re-encoding. Update `configs/project_01.yaml` if
your local source corpus is stored elsewhere.

The ten classes are `airplane`, `bird`, `car`, `cat`, `deer`, `dog`, `horse`, `monkey`,
`ship`, and `truck`. The final corpus contains exactly 30 images per class.

`manifest.csv` is the authoritative list of copied images. `audit.csv` records reviewed
candidates and decisions. `cases.jsonl` expands every accepted image into the four prompt
conditions used by both models.

The image bytes and contact sheets are ignored by Git because source licenses have not
been uniformly verified. Re-run `post-training-evals prepare-data` on a machine with the
source corpus to reconstruct them and verify their SHA-256 hashes.
