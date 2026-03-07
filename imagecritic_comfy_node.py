from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io
import comfy.model_management

# Make the ImageCritic package importable when running inside ComfyUI.
IMAGECRITIC_ROOT = Path(__file__).resolve().parents[2] / "ImageCritic"
if IMAGECRITIC_ROOT.exists() and str(IMAGECRITIC_ROOT) not in sys.path:
    sys.path.append(str(IMAGECRITIC_ROOT))

from src.detail_encoder import DetailEncoder
from src.kontext_custom_pipeline import FluxKontextPipelineWithPhotoEncoderAddTokens
from src.lora_helper import set_single_lora


class _PipelineCache:
    pipeline: FluxKontextPipelineWithPhotoEncoderAddTokens | None = None
    device: torch.device | None = None
    dtype: torch.dtype | None = None
    base_dir: Path | None = None
    detail_path: Path | None = None
    lora_path: Path | None = None


CACHE = _PipelineCache()

# Preferred resolutions copied from the original flux-kontext guidance.
PREFERRED_KONTEXT_RESOLUTIONS: Tuple[Tuple[int, int], ...] = (
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
)


def _pick_kontext_resolution(width: int, height: int) -> tuple[int, int]:
    aspect_ratio = width / height
    _, w, h = min((abs(aspect_ratio - rw / rh), rw, rh) for rw, rh in PREFERRED_KONTEXT_RESOLUTIONS)
    return w, h


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim == 4:
        image = image[0]
    image = image.clamp(0, 1)
    data = (image.cpu().numpy() * 255).astype(np.uint8)
    if data.ndim == 2:
        data = np.stack([data] * 3, axis=-1)
    return Image.fromarray(data, mode="RGB")


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    data = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(data).unsqueeze(0)


def _ensure_weights(models_dir: Path, base_model_dir: Path) -> tuple[Path, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    base_model_dir.mkdir(parents=True, exist_ok=True)

    detail_path = models_dir / "detail_encoder.safetensors"
    lora_path = models_dir / "lora.safetensors"

    if not detail_path.exists():
        hf_hub_download(
            repo_id="ziheng1234/ImageCritic",
            filename=detail_path.name,
            local_dir=str(models_dir),
        )
    if not lora_path.exists():
        hf_hub_download(
            repo_id="ziheng1234/ImageCritic",
            filename=lora_path.name,
            local_dir=str(models_dir),
        )

    # If the kontext base has not been downloaded yet, fetch it once.
    if not (base_model_dir / "transformer").exists():
        snapshot_download(
            repo_id="ziheng1234/kontext",
            local_dir=str(base_model_dir),
            repo_type="model",
            resume_download=True,
            max_workers=8,
        )

    return detail_path, lora_path


def _load_pipeline(
    *,
    base_model_dir: Path,
    models_dir: Path,
    lora_weight: float,
    device: torch.device,
) -> FluxKontextPipelineWithPhotoEncoderAddTokens:
    detail_path, lora_path = _ensure_weights(models_dir, base_model_dir)

    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    needs_reload = (
        CACHE.pipeline is None
        or CACHE.device != device
        or CACHE.dtype != dtype
        or CACHE.base_dir != base_model_dir
    )

    if needs_reload:
        pipeline = FluxKontextPipelineWithPhotoEncoderAddTokens.from_pretrained(
            str(base_model_dir),
            torch_dtype=dtype,
        )
        pipeline.to(device)

        state_dict = load_file(str(detail_path))
        detail_encoder = DetailEncoder().to(dtype=pipeline.transformer.dtype, device=device)
        with torch.no_grad():
            for name, param in detail_encoder.named_parameters():
                if name in state_dict:
                    param.add_(state_dict[name].to(param.device, dtype=param.dtype))

        pipeline.detail_encoder = detail_encoder

        CACHE.pipeline = pipeline
        CACHE.device = device
        CACHE.dtype = dtype
        CACHE.base_dir = base_model_dir
        CACHE.detail_path = detail_path
        CACHE.lora_path = lora_path

    # Refresh LoRA weights on every call in case weight value changes.
    set_single_lora(CACHE.pipeline.transformer, str(lora_path), lora_weights=[lora_weight])
    return CACHE.pipeline


class ImageCriticFluxKontext(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        project_root = IMAGECRITIC_ROOT
        default_models_dir = project_root / "models"
        default_base_dir = project_root / "kontext"

        return io.Schema(
            node_id="ImageCriticFluxKontext",
            display_name="ImageCritic Flux Kontext",
            category="image/refinement",
            description="Runs the ImageCritic flux-kontext pipeline with two reference images.",
            inputs=[
                io.Image.Input("image_a", description="Reference image (IMG1)."),
                io.Image.Input("image_b", description="Target image to refine (IMG2)."),
                io.String.Input("product_tag", default="product"),
                io.String.Input(
                    "prompt_override",
                    default="",
                    multiline=True,
                    dynamic_prompts=True,
                    description="Optional full prompt; leave blank to use the default template.",
                ),
                io.Float.Input("guidance", default=3.5, min=0.0, max=20.0, step=0.1),
                io.Int.Input("seed", default=0, min=0, max=2**31 - 1),
                io.Float.Input("lora_weight", default=1.0, min=0.0, max=2.0, step=0.05),
                io.String.Input(
                    "models_dir",
                    default=str(default_models_dir),
                    description="Folder containing detail_encoder.safetensors and lora.safetensors.",
                ),
                io.String.Input(
                    "base_model_dir",
                    default=str(default_base_dir),
                    description="Folder with the downloaded flux-kontext base model.",
                ),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(
        cls,
        image_a,
        image_b,
        product_tag: str,
        prompt_override: str,
        guidance: float,
        seed: int,
        lora_weight: float,
        models_dir: str,
        base_model_dir: str,
    ) -> io.NodeOutput:
        device = comfy.model_management.get_torch_device()

        pipeline = _load_pipeline(
            base_model_dir=Path(base_model_dir),
            models_dir=Path(models_dir),
            lora_weight=lora_weight,
            device=device,
        )

        pil_a = _tensor_to_pil(image_a)
        pil_b = _tensor_to_pil(image_b)

        target_w, target_h = _pick_kontext_resolution(pil_b.width, pil_b.height)
        resized_a = pil_a.resize((target_w, target_h), Image.Resampling.LANCZOS)
        resized_b = pil_b.resize((target_w, target_h), Image.Resampling.LANCZOS)

        prompt = (
            prompt_override.strip()
            if prompt_override.strip()
            else f"use the {product_tag} in IMG1 as a reference to refine, replace, enhance the {product_tag} in IMG2"
        )

        generator = torch.Generator(device=device).manual_seed(int(seed))
        output = pipeline(
            image_A=resized_a,
            image_B=resized_b,
            prompt=prompt,
            height=target_h,
            width=target_w,
            guidance_scale=float(guidance),
            generator=generator,
        ).images[0]

        output = output.resize((pil_b.width, pil_b.height), Image.Resampling.LANCZOS)
        result_tensor = _pil_to_tensor(output)
        return io.NodeOutput(result_tensor)


class ImageCriticExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ImageCriticFluxKontext]


async def comfy_entrypoint() -> ImageCriticExtension:
    return ImageCriticExtension()
