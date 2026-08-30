"""Qwen loader + JSON-safe generation wrapper.

Loaded ONCE in Colab notebook Cell 4 and reused across every job and every
stage that needs it (research, fact-check, narration, visual bible, scene
planning) -- reloading the model per call would be far too slow. main.py
never imports transformers directly; it only ever calls methods on a
QwenClient instance passed in from the notebook.
"""

from __future__ import annotations
import json
import re

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# Recommendation for a free-tier T4 (14.56GB VRAM): Qwen 3B + SDXL-Lightning
# together total ~13GB even at FULL precision, no quantization required --
# real margin instead of a razor's edge. Qwen 7B needs 4-bit quantization
# working correctly to fit alongside an image model at all, which has been
# an added source of risk in practice. Pass a different model_name to
# QwenClient.load() if you want to try 7B anyway.


class QwenClient:
    def __init__(self, model=None, tokenizer=None, device="cuda"):
        # Kept intentionally simple: use QwenClient.load(...) for model loading.
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._quantized = False  # set True in load() when 4-bit quantization is active

    def offload_to_cpu(self) -> None:
        """Frees Qwen's GPU memory without discarding the loaded weights, so
        it can be restored cheaply later rather than reloaded from disk.
        main.py calls this automatically right before the image_generation
        stage, since Qwen isn't needed again until the next job's research
        stage -- freeing this memory gives the image model real headroom on
        a tight-VRAM GPU instead of Qwen sitting resident and unused.

        Note: bitsandbytes-quantized models generally CANNOT be moved back
        to GPU with a plain .to("cuda") call once offloaded this way -- if
        self._quantized is True, this is a no-op with a warning instead of
        risking a broken model. Quantized Qwen is already only ~2-3GB (3B)
        or ~5-6GB (7B), which is the scenario where offloading matters
        least anyway; it's full-precision Qwen where freeing this memory
        during image generation actually matters, and that case
        offloads/restores safely.
        """
        if self.model is None or self.device == "cpu":
            return
        if self._quantized:
            print("QwenClient.offload_to_cpu(): skipped -- quantized models "
                  "can't be safely moved between devices after loading.")
            return
        import torch
        self.model.to("cpu")
        self.device = "cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def restore_to_gpu(self, device: str = "cuda") -> None:
        """Reverses offload_to_cpu(). No-op if already on GPU or if the
        model is quantized (see offload_to_cpu's note)."""
        if self.model is None or self.device == device or self._quantized:
            return
        self.model.to(device)
        self.device = device

    @classmethod
    def load(cls, model_name: str = DEFAULT_MODEL, device: str = "cuda",
              load_in_4bit: bool = True) -> "QwenClient":
        """Call this once from Colab Cell 4, e.g.:
            from factory.qwen_client import QwenClient
            qwen = QwenClient.load()
            models = {"qwen": qwen}
        Requires a GPU runtime (Runtime -> Change runtime type -> GPU (T4 or
        better)).

        load_in_4bit=True (the default) uses bitsandbytes 4-bit
        quantization, cutting VRAM footprint substantially (e.g. Qwen 7B
        goes from ~14-16GB to ~5-6GB). This mainly matters if you switch to
        a larger checkpoint than the 3B default; 3B is small enough that
        quantization is optional either way.
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
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            quantization_config=quantization_config,
        )

        if device == "cuda":
            # Direct confirmation of whether 4-bit quantization actually
            # took effect, rather than inferring it from download logs.
            allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            print(f"Qwen GPU memory after load: {allocated_gb:.2f} GB "
                  f"({'looks quantized' if allocated_gb < 8 else 'looks like FULL PRECISION -- check bitsandbytes is installed'})")

        client = cls(model=model, tokenizer=tokenizer, device=device)
        client._quantized = bool(quantization_config)
        return client

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
        can't get valid JSON -- callers must catch this and fail/checkpoint
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
