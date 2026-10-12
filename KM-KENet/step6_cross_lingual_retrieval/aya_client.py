"""Chat clients for the Aya multilingual model (local Hugging Face weights or the Cohere API)."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

Message = Mapping[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


class ChatClient(ABC):
    name: str = "chat"

    @abstractmethod
    def chat(self, messages: Sequence[Message], max_new_tokens: int | None = None) -> str:
        """Return the assistant reply for a conversation."""


class HuggingFaceAyaClient(ChatClient):
    """Aya (e.g. CohereLabs/aya-expanse-8b) loaded locally with `transformers`."""

    name = "aya-hf"

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        load_in_4bit: bool = False,
        device_map: str | None = "auto",
        allow_cpu: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        has_cuda = torch.cuda.is_available()
        if not has_cuda and not allow_cpu:
            raise RuntimeError("No CUDA device; set retrieval.aya.allow_cpu_inference=true to run Aya on CPU")

        kwargs: dict[str, Any] = {}
        if has_cuda:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            if device_map:
                kwargs["device_map"] = device_map
            if load_in_4bit:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4"
                )
        else:
            dtype = torch.float32

        logger.info("Loading %s (dtype=%s)", model_name, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)
        if "device_map" not in kwargs:
            self.model.to("cuda" if has_cuda else "cpu")
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._torch = torch

    def chat(self, messages: Sequence[Message], max_new_tokens: int | None = None) -> str:
        inputs = self.tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        generation: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation.update(temperature=self.temperature, top_p=0.9)
        with self._torch.inference_mode():
            output = self.model.generate(**inputs, **generation)
        prompt_length = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True).strip()


class CohereAyaClient(ChatClient):
    """Aya served through the Cohere API (requires an API key)."""

    name = "aya-cohere"

    def __init__(self, model: str, api_key_env: str = "COHERE_API_KEY", max_new_tokens: int = 512,
                 temperature: float = 0.2) -> None:
        import cohere

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Environment variable {api_key_env} is not set")
        self.client = cohere.ClientV2(api_key=api_key)
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def chat(self, messages: Sequence[Message], max_new_tokens: int | None = None) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[dict(message) for message in messages],
            temperature=self.temperature,
            max_tokens=max_new_tokens or self.max_new_tokens,
        )
        parts = [item.text for item in (response.message.content or []) if getattr(item, "type", "text") == "text"]
        return "".join(parts).strip()


def build_chat_client(aya_cfg: Mapping[str, Any]) -> ChatClient | None:
    """Create the configured Aya client; return None (fallback mode) if it cannot be created."""
    backend = str(aya_cfg.get("backend", "hf")).lower()
    if backend not in ("hf", "cohere", "none"):
        raise ValueError(f"Unknown Aya backend '{backend}' (expected hf, cohere or none)")
    if backend == "none":
        return None
    max_new_tokens = int(aya_cfg.get("max_new_tokens", 512))
    temperature = float(aya_cfg.get("temperature", 0.2))
    try:
        if backend == "hf":
            return HuggingFaceAyaClient(
                str(aya_cfg.get("hf_model", "CohereLabs/aya-expanse-8b")),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                load_in_4bit=bool(aya_cfg.get("load_in_4bit", False)),
                device_map=aya_cfg.get("device_map", "auto"),
                allow_cpu=bool(aya_cfg.get("allow_cpu_inference", False)),
            )
        return CohereAyaClient(
            str(aya_cfg.get("cohere_model", "c4ai-aya-expanse-32b")),
            api_key_env=str(aya_cfg.get("cohere_api_key_env", "COHERE_API_KEY")),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    except Exception as exc:  # missing weights, no GPU, no API key, network errors, ...
        logger.warning("Aya backend '%s' unavailable (%s); using lexicon translation and template answers", backend, exc)
        return None
