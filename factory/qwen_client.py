"""Qwen loader + JSON-safe generation wrapper.

Loaded ONCE in Colab notebook Cell 4 and reused across every job and every
stage that needs it (research, fact-check, later narration/scene-planning)
— reloading the model per call would be far too slow and would blow past
Colab's session time on its own. main.py never imports transformers
directly; it only ever calls methods on a QwenClient instance passed in
from the notebook.
"""

from __future__ import annotations
import json
import re

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class QwenClient:
    def __init__(self, model=None, tokenizer=None, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def load(cls, model_name: str = DEFAULT_MODEL, device: str = "cuda",
              load_in_4bit: bool = True) -> "QwenClient":
        """Call this once from Colab Cell 4, e.g.:
            from factory.qwen_client import QwenClient
            qwen = QwenClient.load()
            models = {"qwen": qwen}
        Requires a GPU runtime (Runtime -> Change runtime type -> T4 GPU or
        better).

        VRAM note: Qwen2.5-7B-Instruct needs ~14-16GB VRAM in full bf16 --
        on its own that's most of a free-tier T4's ~15GB, leaving nothing
        for FLUX to share the GPU with. load_in_4bit=True (the default)
        uses bitsandbytes 4-bit quantization, cutting Qwen's footprint to
        roughly 5-6GB with a small, usually acceptable quality tradeoff --
        this is what makes it realistic to run Qwen and FLUX (with
        low_vram=True, see image_engine.load_flux) in the same T4 session.
        Set load_in_4bit=False only if you have a bigger GPU (A100 40GB or
        similar) and want full precision, or drop to a smaller checkpoint
        (Qwen2.5-3B-Instruct / Qwen2.5-1.5B-Instruct) as an alternative to
        quantization if you hit further memory pressure.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        quantization_config = None
        if load_in_4bit and device == "cuda":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device,
            quantization_config=quantization_config,
        )
        return cls(model=model, tokenizer=tokenizer, device=device)

    def generate(self, prompt: str, max_new_tokens: int = 2048, temperature: float = 0.7) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        generated = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate_json(self, prompt: str, max_new_tokens: int = 2048, retries: int = 2):
        """Generates, then extracts/repairs JSON from the response. Models
        routinely wrap JSON in prose or markdown fences even when told not
        to, so this retries with a stricter follow-up instruction rather
        than silently writing malformed data. Raises ValueError if it still
        can't get valid JSON — callers must catch this and fail/checkpoint
        the stage as an error rather than write garbage downstream."""
        last_err = None
        current_prompt = prompt
        for attempt in range(retries + 1):
            raw = self.generate(
                current_prompt, max_new_tokens=max_new_tokens,
                temperature=0.7 if attempt == 0 else 0.2,
            )
            try:
                return extract_json(raw)
            except ValueError as e:
                last_err = e
                current_prompt = (
                    prompt
                    + "\n\nYour previous response was not valid JSON. "
                    "Return ONLY the JSON object or array. No commentary, "
                    "no markdown code fences, no explanation."
                )
        raise ValueError(f"Could not extract valid JSON after {retries + 1} attempts: {last_err}")


def extract_json(text: str):
    """Strips markdown fences and pulls the first {...} or [...] block out
    of otherwise-prose output, then parses it."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as e:
            raise ValueError(f"Found a JSON-like block but failed to parse it: {e}")
    raise ValueError("No JSON object or array found in model output")
