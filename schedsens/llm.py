"""Inference wrapper: one resident 4-bit Llama 3 8B Instruct, batch size 1,
greedy decoding, no gradients.  Plus a persistent response cache keyed by
the exact prompt string so that identical logical calls can be reused (and
counted separately from executed calls)."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class GenResult:
    text: str
    in_tokens: int
    out_tokens: int
    seconds: float


def prompt_hash(prompt: str, max_new_tokens: int) -> str:
    return hashlib.sha256(f"{max_new_tokens}\x00{prompt}".encode()).hexdigest()


class HFModel:
    CHUNKED_PREFILL_ABOVE = 3000   # tokens
    PREFILL_CHUNK = 1024

    def __init__(self, model_id: str, max_input_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.set_grad_enabled(False)
        self.model_id = model_id
        self.max_input_tokens = max_input_tokens
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map={"": 0}, torch_dtype=torch.float16,
            attn_implementation="sdpa")
        self.model.eval()
        # end-of-turn / end-of-text markers present in this vocabulary
        # (Llama 3: <|eot_id|>, <|end_of_text|>; Qwen2.5: <|im_end|>, <|endoftext|>)
        vocab = self.tok.get_vocab()
        self.eos = [vocab[t] for t in ("<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>")
                    if t in vocab]

    def template(self, messages: list[dict]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count(self, prompt: str) -> int:
        return len(self.tok(prompt, add_special_tokens=False)["input_ids"])

    def generate(self, prompt: str, max_new_tokens: int, seed: int) -> GenResult:
        torch = self.torch
        torch.manual_seed(seed)
        enc = self.tok(prompt, add_special_tokens=False, return_tensors="pt").to(0)
        n_in = enc["input_ids"].shape[1]
        if n_in > self.max_input_tokens:
            # never silently truncate; eligibility should prevent this
            raise ValueError(f"prompt has {n_in} tokens > budget {self.max_input_tokens}")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        kw = {}
        if n_in > self.CHUNKED_PREFILL_ABOVE:
            # Long prompts: fill the KV cache in fixed-size chunks so activation
            # memory fits an 8 GB GPU; generate() then processes only the last
            # token. Shorter prompts take the single-pass path unchanged.
            from transformers import DynamicCache
            cache = DynamicCache()
            ids = enc["input_ids"]
            for start in range(0, n_in - 1, self.PREFILL_CHUNK):
                self.model.model(input_ids=ids[:, start:min(start + self.PREFILL_CHUNK, n_in - 1)],
                                 past_key_values=cache, use_cache=True)
            kw["past_key_values"] = cache
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  temperature=None, top_p=None, eos_token_id=self.eos,
                                  pad_token_id=self.eos[0], **kw)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gen = out[0, n_in:].tolist()
        text = self.tok.decode(gen, skip_special_tokens=True)
        del out, enc
        return GenResult(text, n_in, len(gen), dt)

    def info(self) -> dict:
        import torch, transformers, bitsandbytes, accelerate
        cfg = self.model.config
        return {"model_id": self.model_id,
                "model_revision": getattr(cfg, "_commit_hash", None),
                "quantization": cfg.quantization_config.to_dict() if hasattr(cfg, "quantization_config") else None,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "transformers": transformers.__version__, "bitsandbytes": bitsandbytes.__version__,
                "accelerate": accelerate.__version__,
                "gpu": torch.cuda.get_device_name(0)}


class ResponseCache:
    def __init__(self, path: Path | None):
        self.path = path
        self.mem: dict[str, dict] = {}
        if path and path.exists():
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    self.mem[r["hash"]] = r

    def get(self, h: str) -> dict | None:
        return self.mem.get(h)

    def put(self, h: str, res: GenResult) -> None:
        rec = {"hash": h, **asdict(res)}
        self.mem[h] = rec
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")


class Executor:
    """Routes logical calls to the model, optionally reusing identical prompts."""

    def __init__(self, model, cache: ResponseCache | None, reuse: bool):
        self.model, self.cache, self.reuse = model, cache, reuse

    def __call__(self, prompt: str, max_new_tokens: int, seed: int) -> tuple[GenResult, bool]:
        h = prompt_hash(prompt, max_new_tokens)
        if self.reuse and self.cache and (hit := self.cache.get(h)):
            return GenResult(hit["text"], hit["in_tokens"], hit["out_tokens"], hit["seconds"]), True
        res = self.model.generate(prompt, max_new_tokens, seed)
        if self.cache is not None and self.reuse:
            self.cache.put(h, res)
        return res, False
