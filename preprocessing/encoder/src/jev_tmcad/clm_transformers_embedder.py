"""Windows-compatible Qwen3-8B last-token encoder for the official CLM Engine.

The released CLM head expects 4096-dimensional, L2-normalized last-token
Qwen3-8B hidden states. This adapter supplies the same Embedder.embed contract
as the official vLLM-backed embedder without importing vLLM.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np


class TransformersEmbedder:
    def __init__(self, *, model: str, revision: str, device: str = "auto",
                 batch_size: int = 4, max_tokens: int = 2048,
                 cache_size: int = 4096):
        if not revision:
            raise ValueError("Pin the Qwen encoder revision for CLM inference")
        if batch_size < 1 or max_tokens < 1:
            raise ValueError("batch_size and max_tokens must be positive")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install Windows encoder dependencies: `pip install 'transformers>=4.51,<5' accelerate safetensors`.") from exc
        self.torch = torch
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected for CLM but PyTorch cannot see a CUDA GPU")
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) \
            if self.device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        memory = None
        if device == "auto" and self.device.startswith("cuda"):
            reserve = 3 * 1024 ** 3
            memory = {i: max(2 * 1024 ** 3, torch.cuda.mem_get_info(i)[0] - reserve)
                      for i in range(torch.cuda.device_count())}
        self.model = AutoModel.from_pretrained(model, revision=revision, torch_dtype=dtype,
                                               device_map="auto" if device == "auto" else self.device,
                                               max_memory=memory, low_cpu_mem_usage=True)
        self.model.eval()
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.cache_size = cache_size
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        if not texts:
            raise ValueError("CLM cannot embed an empty list")
        unique = list(dict.fromkeys(texts))
        missing = [text for text in unique if text not in self.cache]
        tokens_spent = 0
        for start in range(0, len(missing), self.batch_size):
            chunk = missing[start:start + self.batch_size]
            # CLM's official text recipe uses plain text without special tokens.
            encoded = self.tokenizer([text or " " for text in chunk],
                                     add_special_tokens=False, padding=False,
                                     truncation=False)["input_ids"]
            lengths = [len(ids) for ids in encoded]
            if any(n == 0 or n > self.max_tokens for n in lengths):
                raise ValueError("CLM text is empty or exceeds the configured context; refuse silent truncation")
            tokens_spent += sum(lengths)
            padded = self.tokenizer.pad({"input_ids": encoded}, padding=True,
                                        return_tensors="pt")
            input_ids = padded["input_ids"].to(self.device)
            attention_mask = padded["attention_mask"].to(self.device)
            with self.torch.inference_mode():
                states = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                    use_cache=False).last_hidden_state
                last_index = attention_mask.to(states.device).sum(dim=1) - 1
                last = states[self.torch.arange(len(chunk), device=states.device), last_index]
                vectors = self.torch.nn.functional.normalize(last.float(), dim=-1).cpu().numpy()
            if vectors.shape[1] != 4096 or not np.isfinite(vectors).all():
                raise ValueError("Qwen encoder did not produce finite 4096-dimensional vectors")
            for text, vector in zip(chunk, vectors):
                self.cache[text] = vector
                self.cache.move_to_end(text)
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        result = []
        for text in texts:
            self.cache.move_to_end(text)
            result.append(self.cache[text])
        return np.stack(result), tokens_spent

    def close(self) -> None:
        self.cache.clear()
        del self.model
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
