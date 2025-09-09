from torch import Tensor, nn
from transformers import (CLIPTextModel, CLIPTokenizer, T5EncoderModel,
                          T5Tokenizer)
import os
import json


class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        # Check if it's a CLIP model by looking at the tokenizer config
        tokenizer_config_path = os.path.join(version, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, 'r') as f:
                tokenizer_config = json.load(f)
            self.is_clip = tokenizer_config.get("tokenizer_class") == "CLIPTokenizer"
        else:
            # Fallback to original logic
            self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            # Remove torch_dtype from hf_kwargs for tokenizer
            model_kwargs = {k: v for k, v in hf_kwargs.items() if k != 'torch_dtype'}
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **model_kwargs)
        else:
            # Remove torch_dtype from hf_kwargs for tokenizer
            model_kwargs = {k: v for k, v in hf_kwargs.items() if k != 'torch_dtype'}
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version)
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **model_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        # ensure module is on a real device (ZeRO-3 partition may put params on meta/cpu)
        device = next(self.hf_module.parameters()).device
        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
