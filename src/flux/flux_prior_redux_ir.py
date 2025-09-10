# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional, Union

import torch
from PIL import Image
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
    T5EncoderModel,
    T5TokenizerFast,
)
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FluxLoraLoaderMixin,          # If symbol missing, upgrade diffusers or fallback to LoraLoaderMixin
    TextualInversionLoaderMixin,
)
from diffusers.utils import logging
from diffusers.utils.doc_utils import replace_example_docstring
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder
from diffusers.pipelines.flux.pipeline_output import FluxPriorReduxPipelineOutput
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


import torch
import torch.nn.functional as F

def siglip_from_unit_tensor(
    x: torch.Tensor,
    size=(512, 512),
    device=None,
    out_dtype=torch.bfloat16,
    strict_range=True,
):
    """
    x: (1,3,H,W) or (3,H,W), values in [0, 1]
    Returns: (1,3,512,512), values in [-1, 1], equivalent to SiglipImageProcessor

    Key fixes:
    1. Use PIL resize (see resample below) instead of torch bilinear
    2. Ensure exact parity with transformers' SiglipImageProcessor
    """
    from PIL import Image
    import numpy as np
    
    # Support (3,H,W) input
    if x.ndim == 3:
        x = x.unsqueeze(0)
    assert x.ndim == 4 and x.shape[1] == 3, "Expected shape (B,3,H,W)"
    
    if strict_range:
        minv, maxv = float(x.min()), float(x.max())
        if minv < -1e-3 or maxv > 1 + 1e-3:
            raise ValueError(f"Input must be in [0,1], got range [{minv:.4f},{maxv:.4f}]")

    batch_size = x.shape[0]
    processed_batch = []
    
    for i in range(batch_size):
        # 1) Convert to PIL Image (HWC, uint8)
        img_tensor = x[i].permute(1, 2, 0)  # CHW -> HWC
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np, mode='RGB')

        # 2) Resize with PIL (resample=2 i.e., BILINEAR, matches SiglipImageProcessor)
        if pil_img.size != size:
            # resample=2 corresponds to PIL.Image.BILINEAR
            pil_img = pil_img.resize(size, resample=2)

        # 3) Back to tensor and normalize
        img_array = np.array(pil_img, dtype=np.float32) / 255.0  # [0,1]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW

        # 4) Normalize to [-1,1]: (x - 0.5) / 0.5
        img_tensor = (img_tensor - 0.5) / 0.5

        processed_batch.append(img_tensor)
    
    # 5) Stack into batch
    result = torch.stack(processed_batch, dim=0)
    
    # 6) Move to target device/dtype
    if device is not None:
        result = result.to(device=device, dtype=out_dtype)
    else:
        result = result.to(dtype=out_dtype)
    
    return result


class FluxPriorReduxPipeline(DiffusionPipeline):
    r"""
    The Flux Redux pipeline for image-to-image generation.

    Reference: https://blackforestlabs.ai/flux-1-tools/

    Args:
        image_encoder ([`SiglipVisionModel`]):
            SIGLIP vision model to encode the input image.
        feature_extractor ([`SiglipImageProcessor`]):
            Image processor for preprocessing images for the SIGLIP model.
        image_embedder ([`ReduxImageEncoder`]):
            Redux image encoder to process the SIGLIP embeddings.
        text_encoder ([`CLIPTextModel`], *optional*):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`], *optional*):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`, *optional*):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`, *optional*):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    model_cpu_offload_seq = "image_encoder->image_embedder"
    _optional_components = [
        "text_encoder",
        "tokenizer",
        "text_encoder_2",
        "tokenizer_2",
    ]
    _callback_tensor_inputs = []

    def __init__(
        self,
        image_encoder: SiglipVisionModel,
        feature_extractor: SiglipImageProcessor,
        image_embedder: ReduxImageEncoder,
        text_encoder: CLIPTextModel = None,
        tokenizer: CLIPTokenizer = None,
        text_encoder_2: T5EncoderModel = None,
        tokenizer_2: T5TokenizerFast = None,
    ):
        super().__init__()

        self.register_modules(
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            image_embedder=image_embedder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )

    def check_inputs(
        self,
        image,
        prompt,
        prompt_2,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        prompt_embeds_scale=1.0,
        pooled_prompt_embeds_scale=1.0,
        image_latents=None,
        image_embeds=None,
    ):
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")
        if prompt is not None and (isinstance(prompt, list) and isinstance(image, list) and len(prompt) != len(image)):
            raise ValueError(
                f"number of prompts must be equal to number of images, but {len(prompt)} prompts were provided and {len(image)} images"
            )
        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )
        # Infer batch size for consistency checks
        inferred_batch_size = None
        if image_embeds is not None and isinstance(image_embeds, torch.Tensor):
            inferred_batch_size = image_embeds.shape[0]
        elif image_latents is not None and isinstance(image_latents, torch.Tensor):
            inferred_batch_size = image_latents.shape[0] if image_latents.dim() >= 3 else 1
        elif image is not None:
            if isinstance(image, torch.Tensor):
                inferred_batch_size = image.shape[0] if image.dim() == 4 else 1
            elif isinstance(image, dict) and "pixel_values" in image and isinstance(image["pixel_values"], torch.Tensor):
                inferred_batch_size = image["pixel_values"].shape[0] if image["pixel_values"].dim() == 4 else 1
            elif isinstance(image, list):
                inferred_batch_size = len(image)
            elif isinstance(image, Image.Image):
                inferred_batch_size = 1

        if isinstance(prompt, list) and inferred_batch_size is not None and len(prompt) != inferred_batch_size:
            raise ValueError(
                f"number of prompts must be equal to batch size, but {len(prompt)} prompts were provided and batch size is {inferred_batch_size}"
            )

        if isinstance(prompt_embeds_scale, list) and inferred_batch_size is not None and len(prompt_embeds_scale) != inferred_batch_size:
            raise ValueError(
                f"number of weights must be equal to batch size, but {len(prompt_embeds_scale)} weights were provided and batch size is {inferred_batch_size}"
            )

    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype

        # Support three input categories:
        # 1) Preprocessed pixel tensor: torch.Tensor[B,3,H,W] or [3,H,W]
        # 2) Dict/BatchFeature containing 'pixel_values'
        # 3) Raw PIL/np/list to be preprocessed by feature_extractor
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            inputs = {"pixel_values": image.to(device=device, dtype=dtype)}
        elif isinstance(image, dict):
            if "pixel_values" in image:
                pixel_values = image["pixel_values"]
                if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 3:
                    pixel_values = pixel_values.unsqueeze(0)
                inputs = {"pixel_values": pixel_values.to(device=device, dtype=dtype)}
            else:
                # If it's a transformers BatchFeature, it usually has .to()
                if hasattr(image, "to"):
                    image = image.to(device=device, dtype=dtype)
                    inputs = image
                else:
                    raise ValueError("Unsupported dict input for `image`. Expect key 'pixel_values' or a BatchFeature with .to().")
        else:
            inputs = self.feature_extractor.preprocess(
                images=image, do_resize=True, return_tensors="pt", do_convert_rgb=True
            )
            inputs = inputs.to(device=device, dtype=dtype)

        image_enc_hidden_states = self.image_encoder(**inputs).last_hidden_state
        image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)

        return image_enc_hidden_states

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._get_clip_prompt_embeds
    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale


        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_embeds_scale: Optional[Union[float, List[float]]] = 1.0,
        pooled_prompt_embeds_scale: Optional[Union[float, List[float]]] = 1.0,
        image_latents: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        aggregate_batch: bool = False,
        offload: bool = False,
        return_dict: bool = True,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)`
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. **experimental feature**: to use this feature,
                make sure to explicitly load text encoders to the pipeline. Prompts will be ignored if text encoders
                are not loaded.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPriorReduxPipelineOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPriorReduxPipelineOutput`] or `tuple`:
            [`~pipelines.flux.FluxPriorReduxPipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is a list with the generated images.
        """

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            image,
            prompt,
            prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_embeds_scale=prompt_embeds_scale,
            pooled_prompt_embeds_scale=pooled_prompt_embeds_scale,
            image_latents=image_latents,
            image_embeds=image_embeds,
        )

        # 2. Define call parameters
        if image_embeds is not None:
            if isinstance(image_embeds, torch.Tensor):
                if image_embeds.dim() == 3:
                    batch_size = image_embeds.shape[0]
                elif image_embeds.dim() == 2:
                    batch_size = 1
                else:
                    raise ValueError("`image_embeds` must be 2D or 3D tensor.")
            else:
                raise ValueError("`image_embeds` must be a torch.Tensor.")
        elif image_latents is not None:
            batch_size = image_latents.shape[0] if image_latents.dim() >= 3 else 1
        elif image is not None and isinstance(image, Image.Image):
            batch_size = 1
        elif image is not None and isinstance(image, dict):
            if "pixel_values" in image and isinstance(image["pixel_values"], torch.Tensor):
                pixel_values = image["pixel_values"]
                batch_size = pixel_values.shape[0] if pixel_values.dim() == 4 else 1
            else:
                # transformers BatchFeature typically contains 'pixel_values'
                batch_size = 1
        elif image is not None and isinstance(image, torch.Tensor):
            batch_size = image.shape[0] if image.dim() == 4 else 1
        elif image is not None and isinstance(image, list):
            batch_size = len(image)
        else:
            raise ValueError("One of `image`, `image_latents`, or `image_embeds` must be provided.")
        if prompt is not None and isinstance(prompt, str):
            prompt = batch_size * [prompt]
        if isinstance(prompt_embeds_scale, float):
            prompt_embeds_scale = batch_size * [prompt_embeds_scale]
        if isinstance(pooled_prompt_embeds_scale, float):
            pooled_prompt_embeds_scale = batch_size * [pooled_prompt_embeds_scale]

        device = self._execution_device

        # 3. Prepare image embeddings (allow passing image_latents to bypass the vision encoder)
        if image_embeds is None:
            if image_latents is None:
                image_latents = self.encode_image(image, device, 1)
            else:
                # Align device and dtype
                dtype_image_encoder = next(self.image_encoder.parameters()).dtype
                if image_latents.dim() == 2:
                    # Unexpected shape; expect at least (B, N, D)
                    raise ValueError("`image_latents` shape is invalid. Expect at least 3D tensor (B, N, D).")
                image_latents = image_latents.to(device=device, dtype=dtype_image_encoder)

            image_embeds = self.image_embedder(image_latents).image_embeds
            image_embeds = image_embeds.to(device=device)
        else:
            image_embeds = image_embeds.to(device=device)
            if image_embeds.dim() == 2:
                image_embeds = image_embeds.unsqueeze(1)

        # 3. Prepare text embeddings (allow passing prompt_embeds/pooled_prompt_embeds directly)
        if prompt_embeds is not None and pooled_prompt_embeds is not None:
            # Normalize shapes and devices
            if isinstance(prompt_embeds, torch.Tensor) and prompt_embeds.dim() == 2:
                prompt_embeds = prompt_embeds.unsqueeze(0)
            if isinstance(pooled_prompt_embeds, torch.Tensor) and pooled_prompt_embeds.dim() == 1:
                pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(0)
            prompt_embeds = prompt_embeds.to(device=device, dtype=image_embeds.dtype)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=image_embeds.dtype)
        else:
            if hasattr(self, "text_encoder") and self.text_encoder is not None:
                (
                    prompt_embeds,
                    pooled_prompt_embeds,
                    _,
                ) = self.encode_prompt(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    device=device,
                    num_images_per_prompt=1,
                    max_sequence_length=512,
                    lora_scale=None,
                )
            else:
                if prompt is not None:
                    logger.warning(
                        "prompt input is ignored when text encoders are not loaded to the pipeline. "
                        "Make sure to explicitly load the text encoders to enable prompt input. "
                    )
                # max_sequence_length is 512, t5 encoder hidden size is 4096
                prompt_embeds = torch.zeros((batch_size, 512, 4096), device=device, dtype=image_embeds.dtype)
                # pooled_prompt_embeds is 768, clip text encoder hidden size
                pooled_prompt_embeds = torch.zeros((batch_size, 768), device=device, dtype=image_embeds.dtype)

        # Scale and concatenate image/text embeddings (validate last dim equality)
        if prompt_embeds.shape[-1] != image_embeds.shape[-1]:
            raise ValueError(
                f"The last dimension of `prompt_embeds` ({prompt_embeds.shape[-1]}) must match that of `image_embeds` ({image_embeds.shape[-1]})."
            )
        prompt_embeds = torch.cat([prompt_embeds, image_embeds], dim=1)

        prompt_embeds *= torch.tensor(prompt_embeds_scale, device=device, dtype=image_embeds.dtype)[:, None, None]
        pooled_prompt_embeds *= torch.tensor(pooled_prompt_embeds_scale, device=device, dtype=image_embeds.dtype)[
            :, None
        ]

        # Weighted sum across batch or keep per-sample
        if aggregate_batch:
            prompt_embeds = torch.sum(prompt_embeds, dim=0, keepdim=True)
            pooled_prompt_embeds = torch.sum(pooled_prompt_embeds, dim=0, keepdim=True)

        # Offload all models
        if offload:
            self.maybe_free_model_hooks()

        if not return_dict:
            return (prompt_embeds, pooled_prompt_embeds)

        return FluxPriorReduxPipelineOutput(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds)
