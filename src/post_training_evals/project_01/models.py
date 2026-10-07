from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class LoadedVLM:
    model_id: str
    revision: str
    model: Any
    processor: Any
    torch: Any
    device: Any

    @classmethod
    def load(
        cls,
        model_id: str,
        revision: str,
        dtype_name: str = "bfloat16",
    ) -> "LoadedVLM":
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU fallback is intentionally disabled")
        dtype = getattr(torch, dtype_name)
        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            revision=revision,
            dtype=dtype,
            device_map={"": 0},
            attn_implementation="sdpa",
        ).eval()
        device = next(model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(f"Model loaded on {device}, expected CUDA")
        resolved_revision = str(getattr(model.config, "_commit_hash", None) or revision)
        return cls(
            model_id=model_id,
            revision=resolved_revision,
            model=model,
            processor=processor,
            torch=torch,
            device=device,
        )

    def generate(self, image: Any, prompt: str, *, max_new_tokens: int) -> tuple[str, dict[str, int]]:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        rendered = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=rendered, images=[image], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        input_tokens = int(inputs["input_ids"].shape[-1])
        self.torch.cuda.reset_peak_memory_stats(self.device)
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        generated = outputs[0, input_tokens:]
        text = self.processor.decode(generated, skip_special_tokens=True).strip()
        stats = {
            "input_tokens": input_tokens,
            "output_tokens": int(generated.shape[-1]),
            "peak_vram_bytes": int(self.torch.cuda.max_memory_allocated(self.device)),
        }
        return text, stats
