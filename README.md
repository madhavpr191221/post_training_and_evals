# Post-Training and Evaluations

Research-engineering exercises that pair changes in model behavior with explicit evaluation.

## Project 1: base versus instruct

Project 1 is an inference-only comparison of:

- `HuggingFaceTB/SmolVLM-Base`
- `HuggingFaceTB/SmolVLM-Instruct`

The evaluation uses a visually audited, balanced set of 300 natural images copied from the
local `perception_world_models` image corpus. No training or parameter updates occur.

The copied image bytes remain local and are ignored by Git because their upstream licenses
have not been uniformly verified. The tracked manifest retains provenance and hashes;
credential-like signed query parameters are removed from source URLs before publication.

### Commands

```powershell
uv run post-training-evals prepare-data
uv run post-training-evals run --model base
uv run post-training-evals run --model instruct
uv run post-training-evals report
```

See `data/project_01/README.md` for the dataset protocol and
`reports/project_01_baseline.md` for results after both inference runs complete.
