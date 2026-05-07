from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


@dataclass
class DirectionBundle:
    emotions: list[str]
    layers: list[int]
    vectors: np.ndarray
    neutral_mean: np.ndarray

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            emotions=np.array(self.emotions),
            layers=np.array(self.layers, dtype=np.int32),
            vectors=self.vectors,
            neutral_mean=self.neutral_mean,
        )


def load_model_and_tokenizer(model_config: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = model_config["name"]
    revision = model_config.get("revision")
    device_name = model_config.get("device", "auto")
    dtype_name = model_config.get("dtype", "auto")
    tokenizer_backend = model_config.get("tokenizer_backend", "auto")
    dtype = _torch_dtype(dtype_name, torch)
    if tokenizer_backend == "mistral_common":
        tokenizer = _load_mistral_common_tokenizer(name, revision=revision)
    else:
        tokenizer_kwargs = {"trust_remote_code": True}
        if revision:
            tokenizer_kwargs["revision"] = revision
        tokenizer = AutoTokenizer.from_pretrained(name, **tokenizer_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"trust_remote_code": True}
    if revision:
        kwargs["revision"] = revision
    if torch.cuda.is_available() and device_name == "auto":
        kwargs["device_map"] = "auto"
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
    else:
        if dtype is not None:
            kwargs["torch_dtype"] = dtype

    _install_transformers_remote_code_compat()
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if torch.cuda.is_available() and device_name == "cuda":
        model.to("cuda")
    elif not torch.cuda.is_available():
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
    model.eval()
    return model, tokenizer


class MistralCommonTokenizerAdapter:
    """Small adapter for Mistral's official tokenizer API."""

    uses_mistral_common = True
    chat_template = True

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer
        self._base_tokenizer = tokenizer.instruct_tokenizer.tokenizer
        self.bos_token_id = self._base_tokenizer.bos_id
        self.eos_token_id = self._base_tokenizer.eos_id
        pad_token_id = self._base_tokenizer.pad_id
        self.pad_token_id = pad_token_id if pad_token_id is not None else self.eos_token_id
        self.pad_token = "<pad>"
        self.eos_token = "</s>"

    def __len__(self) -> int:
        return int(self._base_tokenizer.n_words)

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str = "pt",
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, Any]:
        tokens = self._base_tokenizer.encode(text, bos=True, eos=False)
        return self._tensor_batch(tokens, return_tensors, truncation, max_length)

    def encode_chat_prompt(
        self,
        text: str,
        *,
        return_tensors: str = "pt",
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, Any]:
        from mistral_common.protocol.instruct.messages import UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest

        request = ChatCompletionRequest(messages=[UserMessage(content=text)])
        tokens = self._tokenizer.encode_chat_completion(request).tokens
        return self._tensor_batch(tokens, return_tensors, truncation, max_length)

    def decode(self, token_ids, *, skip_special_tokens: bool = True) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        decoded = self._tokenizer.decode([int(token_id) for token_id in token_ids])
        if skip_special_tokens:
            return decoded.strip()
        return decoded

    def _tensor_batch(
        self,
        tokens: list[int],
        return_tensors: str,
        truncation: bool,
        max_length: int | None,
    ) -> dict[str, Any]:
        if truncation and max_length is not None:
            tokens = tokens[-max_length:]
        if return_tensors != "pt":
            return {"input_ids": [tokens], "attention_mask": [[1] * len(tokens)]}

        import torch

        input_ids = torch.tensor([tokens], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _load_mistral_common_tokenizer(name: str, revision: str | None = None) -> MistralCommonTokenizerAdapter:
    try:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    except ImportError as exc:
        raise ImportError(
            "mistral-common is required for model.tokenizer_backend='mistral_common'. "
            "Install it with `pip install mistral-common>=1.7.0`."
        ) from exc

    return MistralCommonTokenizerAdapter(MistralTokenizer.from_hf_hub(name, revision=revision))


def _install_transformers_remote_code_compat() -> None:
    try:
        import torch
        from transformers.cache_utils import DynamicCache
        from transformers.utils import import_utils
    except ImportError:
        return

    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: hasattr(torch, "fx")
    if not hasattr(DynamicCache, "from_legacy_cache"):

        @classmethod
        def from_legacy_cache(cls, past_key_values=None):
            if past_key_values is None:
                return cls()
            return cls(ddp_cache_data=past_key_values)

        DynamicCache.from_legacy_cache = from_legacy_cache
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, _new_length=None: self.get_seq_length()
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "to_legacy_cache"):
        DynamicCache.to_legacy_cache = lambda self: tuple(
            (layer.keys, layer.values) for layer in self.layers
        )


def _torch_dtype(dtype_name: str, torch_module):
    if dtype_name in (None, "auto"):
        return None
    mapping = {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def find_transformer_blocks(model) -> list:
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for parent_name, child_name in candidates:
        parent = getattr(model, parent_name, None)
        if parent is not None and hasattr(parent, child_name):
            return list(getattr(parent, child_name))
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        decoder = model.model.decoder
        if hasattr(decoder, "layers"):
            return list(decoder.layers)
    raise ValueError("could not find transformer blocks for this model")


def select_layers(num_layers: int, spec) -> list[int]:
    if spec == "auto":
        raw = [num_layers // 3, num_layers // 2, (2 * num_layers) // 3]
        layers = sorted({layer for layer in raw if 0 <= layer < num_layers})
    elif isinstance(spec, list):
        raw = [int(layer) for layer in spec]
        invalid = [layer for layer in raw if layer < 0 or layer >= num_layers]
        if invalid:
            raise ValueError(f"invalid layer(s) for {num_layers} layers: {invalid}")
        layers = sorted(set(raw))
    else:
        raw = [int(part) for part in str(spec).split(",")]
        invalid = [layer for layer in raw if layer < 0 or layer >= num_layers]
        if invalid:
            raise ValueError(f"invalid layer(s) for {num_layers} layers: {invalid}")
        layers = sorted(set(raw))
    if not layers:
        raise ValueError(f"no valid layers selected from {spec!r}")
    return layers


def mean_activations(model, tokenizer, texts: list[str], layers: list[int]) -> np.ndarray:
    import torch

    device = next(model.parameters()).device
    vocab_size = _model_vocab_size(model)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for text in texts:
            encoded = _encode_prompt(
                tokenizer,
                text,
                chat=False,
                truncation=True,
                max_length=1024,
            )
            _validate_token_ids(encoded, vocab_size)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            result = model(**encoded, output_hidden_states=True, use_cache=False)
            mask = encoded["attention_mask"].bool()[0]
            layer_vectors = []
            for layer in layers:
                hidden = result.hidden_states[layer + 1][0, mask, :]
                if hidden.shape[0] > 1:
                    hidden = hidden[1:, :]
                layer_vectors.append(hidden.float().mean(dim=0).cpu().numpy())
            outputs.append(np.stack(layer_vectors, axis=0))
    return np.stack(outputs, axis=0)


def build_direction_bundle(
    model,
    tokenizer,
    snippets_by_emotion: dict[str, list[str]],
    neutral_snippets: list[str],
    layers: list[int],
) -> DirectionBundle:
    neutral = mean_activations(model, tokenizer, neutral_snippets, layers)
    neutral_mean = neutral.mean(axis=0)
    emotions = list(snippets_by_emotion)
    vectors = []
    for emotion in emotions:
        acts = mean_activations(model, tokenizer, snippets_by_emotion[emotion], layers)
        vectors.append(acts.mean(axis=0) - neutral_mean)
    return DirectionBundle(
        emotions=emotions,
        layers=layers,
        vectors=np.stack(vectors, axis=0),
        neutral_mean=neutral_mean,
    )


def activation_scores(
    activations: np.ndarray,
    bundle: DirectionBundle,
) -> np.ndarray:
    centered = activations[:, None, :, :] - bundle.neutral_mean[None, None, :, :]
    directions = bundle.vectors[None, :, :, :]
    numerator = (centered * directions).sum(axis=-1)
    centered_norm = np.linalg.norm(centered, axis=-1)
    direction_norm = np.linalg.norm(directions, axis=-1)
    return numerator / np.maximum(centered_norm * direction_norm, 1e-8)


def format_user_prompt(tokenizer, text: str) -> str:
    if getattr(tokenizer, "uses_mistral_common", False):
        return text
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return text


def generate_text(
    model,
    tokenizer,
    prompt: str,
    generation_config: dict,
    bundle: DirectionBundle | None = None,
    emotion: str | None = None,
    strength: float = 0.0,
    seed: int | None = None,
) -> str:
    import torch

    device = next(model.parameters()).device
    encoded = _encode_prompt(tokenizer, prompt, chat=True, truncation=True, max_length=2048)
    _validate_token_ids(encoded, _model_vocab_size(model))
    encoded = {key: value.to(device) for key, value in encoded.items()}
    max_new_tokens = int(generation_config.get("max_new_tokens", 160))
    temperature = float(generation_config.get("temperature", 0.2))
    top_p = float(generation_config.get("top_p", 0.9))
    do_sample = temperature > 0
    generate_kwargs = {
        **encoded,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    with steering(model, bundle, emotion, strength):
        with torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            generated = model.generate(**generate_kwargs)
    new_tokens = generated[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _encode_prompt(
    tokenizer,
    text: str,
    *,
    chat: bool,
    truncation: bool,
    max_length: int,
) -> dict[str, Any]:
    if chat and hasattr(tokenizer, "encode_chat_prompt"):
        return tokenizer.encode_chat_prompt(
            text,
            return_tensors="pt",
            truncation=truncation,
            max_length=max_length,
        )
    formatted = format_user_prompt(tokenizer, text) if chat else text
    return tokenizer(
        formatted,
        return_tensors="pt",
        truncation=truncation,
        max_length=max_length,
    )


def _model_vocab_size(model) -> int:
    return int(model.get_input_embeddings().weight.shape[0])


def _validate_token_ids(encoded: dict[str, Any], vocab_size: int) -> None:
    input_ids = encoded["input_ids"]
    max_id = int(input_ids.max().item())
    if max_id >= vocab_size:
        raise ValueError(
            f"tokenizer produced token id {max_id}, but model embeddings only "
            f"cover ids < {vocab_size}. Use the model's official tokenizer backend."
        )


@contextmanager
def steering(
    model,
    bundle: DirectionBundle | None,
    emotion: str | None,
    strength: float,
) -> Iterator[None]:
    if bundle is None or emotion is None or strength == 0:
        yield
        return

    import torch

    blocks = find_transformer_blocks(model)
    emotion_index = bundle.emotions.index(emotion)
    handles = []
    layer_to_vector = {
        layer: torch.tensor(bundle.vectors[emotion_index, idx])
        for idx, layer in enumerate(bundle.layers)
    }

    def make_hook(layer: int):
        direction = layer_to_vector[layer]

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            delta = direction.to(device=hidden.device, dtype=hidden.dtype)
            delta = delta / torch.clamp(delta.norm(), min=1e-6)
            residual_norm = hidden.detach().float().norm(dim=-1).mean().to(hidden.dtype)
            steered = hidden + (0.1 * float(strength) * residual_norm * delta)
            if isinstance(output, tuple):
                return (steered, *output[1:])
            return steered

        return hook

    for layer in bundle.layers:
        handles.append(blocks[layer].register_forward_hook(make_hook(layer)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
