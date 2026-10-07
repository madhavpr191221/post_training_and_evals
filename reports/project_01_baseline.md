# Project 1: SmolVLM Base vs Instruct

This is an inference-only paired comparison: the two checkpoints were evaluated on identical image-prompt cases; no training or parameter updates occurred.

## Evaluation setup

- Base: `HuggingFaceTB/SmolVLM-Base` at revision `a576ab7b4354916a0cb5b58a3bd973a9a526ee7c`
- Instruct: `HuggingFaceTB/SmolVLM-Instruct` at revision `81cd9a775a4d644f2faf4e7becff4559b46b14c7`
- Dataset: 300 visually reviewed web-search images, balanced at 30 images in each of ten folder-label classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck.
- Selected source files were copied without resizing or re-encoding; image dimensions therefore vary. Folder labels were based on search-query curation and visual review, not independent annotation.
- Each image was evaluated with all four prompts, producing 1,200 image-prompt cases per model. A case means one image paired with one prompt; the four cases from the same image are related, not four independent images.
- Generation used `cuda` with `bfloat16`, deterministic decoding (`do_sample=false`), and a maximum of 48 new tokens.
- The processor formatted each request as a user message containing the image and prompt. Model responses were decoded and stripped of surrounding whitespace before grading.

## Prompts

The prompt suite varies both how constrained the answer is and whether a machine-readable format is requested:

| Condition | Exact prompt | Purpose |
|---|---|---|
| `open` | What is the primary subject of this image? | Open-ended identification, without a supplied label list. |
| `closed_set` | Classify the primary subject of this image as exactly one of: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck. | Choose a class from the ten listed labels, while allowing a natural-language response. |
| `strict_label` | Return only one lowercase class label from this list and no other text: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck. | Test exact single-label output: one lowercase class token and no other text. |
| `json_label` | Return exactly {"label":"<class>"} with no additional keys or text, where <class> is one of: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck. | Test exact structured output with one `label` key and no extra text or keys. |

## What was measured and how responses were scored

- **Semantic accuracy** is the fraction of successful cases whose response could be mapped to exactly one expected class; failed inference cases are excluded from this denominator. The grader lowercases and normalizes text, then recognizes the class names and configured aliases (for example, `aircraft` for airplane and `puppy` for dog). A response mentioning multiple recognized classes, or none, is not mapped to a class. This is a heuristic text grader, not human judgment of visual understanding.
- **Format compliance** is separate from semantic accuracy. For `strict_label`, the stripped response must exactly equal one lowercase class token. For `json_label`, it must parse as a JSON object with exactly one `label` key whose value is exactly one of the ten lowercase class tokens. Open and closed-set prompts have no format target and are excluded from this measure; the overall compliance denominator is therefore the successful strict-label plus JSON cases (600 per model here).
- Consequently, a semantically correct answer can fail format compliance: `Cat.` can be correct for the cat class but fail strict-label formatting, and plain text `cat` can be semantically correct on the JSON prompt while failing JSON formatting. For malformed JSON, the grader still attempts to find a semantic class in the raw response; JSON accuracy is therefore not JSON validity.
- **Overall accuracy** pools the four prompt conditions (all image-prompt cases); it is not an estimate from one prompt or a prompt-independent measure of visual recognition. Prompt-level accuracy is shown separately below.
- **Mean words** is the arithmetic mean of whitespace-separated words in decoded responses. **Median latency** is the median per-case wall time, including image opening/conversion, processor preparation, generation, CUDA synchronization, decoding, and grading; it is not pure model decoding time and excludes writing the result row to disk. **Peak VRAM** is the largest per-generation PyTorch CUDA memory allocation reported across cases, not total device memory usage.

## Overall results

| Model | Cases | Successful | Semantic accuracy | Strict/JSON format compliance | Mean words | Median latency (s) | Peak allocated VRAM (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 1200 | 1200 | 20.2% | 0.0% | 4.18 | 1.359 | 4771.0 |
| instruct | 1200 | 1200 | 92.8% | 0.0% | 2.33 | 1.152 | 4771.0 |

## Paired outcomes

For each shared image-prompt case, compare whether each model was semantically correct. The categories below are mutually exclusive and partition the shared cases:

| Outcome | Cases | Share | Meaning |
|---|---:|---:|---|
| Both Correct | 234 | 19.5% | Both models were judged semantically correct. |
| Base Only | 8 | 0.7% | Base was correct; Instruct was not. |
| Instruct Only | 879 | 73.2% | Instruct was correct; Base was not. |
| Both Wrong | 79 | 6.6% | Neither model was judged correct. |

These counts imply 242 semantically correct Base cases and 1113 semantically correct Instruct cases among 1200 paired cases. In the 887 cases where correctness differed, Instruct was correct on 879 and Base on 8. The outcome shares can differ from 100% by a tenth of a percentage point because of rounding.

The paired cases are not independent image samples: each image contributes four prompt conditions. These counts describe this image set and prompt suite; they do not identify the cause of the model difference or establish performance on other images or prompts.

## Accuracy and format compliance by prompt

| Prompt | Base accuracy | Instruct accuracy | Base format | Instruct format |
|---|---:|---:|---:|---:|
| open | 66.7% | 89.0% | n/a | n/a |
| closed_set | 7.0% | 97.0% | n/a | n/a |
| strict_label | 7.0% | 94.7% | 0.0% | 0.0% |
| json_label | 0.0% | 90.3% | 0.0% | 0.0% |

## Accuracy by class

Each class row pools its 30 images across four prompts (120 cases per model).
| Class | Base | Instruct |
|---|---:|---:|
| airplane | 16.7% | 100.0% |
| bird | 20.0% | 89.2% |
| car | 14.2% | 99.2% |
| cat | 33.3% | 88.3% |
| deer | 16.7% | 100.0% |
| dog | 27.5% | 94.2% |
| horse | 15.0% | 95.0% |
| monkey | 20.8% | 88.3% |
| ship | 21.7% | 97.5% |
| truck | 15.8% | 75.8% |

## Model behavior notes

Instruct usually returned a class label, but its strict-label outputs often included capitalization or punctuation, and its JSON-prompt outputs did not parse as valid JSON. Base frequently emitted empty or image-grid placeholder text under constrained prompts. These are observed response patterns; semantic correctness and requested-format compliance remain distinct outcomes.

## Interpretation boundary

These descriptive results apply to this audited image set and prompt suite. The source labels are coarse object classes and may not capture every reasonable interpretation of an image; review was not independent annotation. The corpus is a curated web-search sample rather than a probability sample. These results do not establish a causal or population-level effect of instruction tuning.
