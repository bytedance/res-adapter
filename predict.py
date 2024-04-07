# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
from typing import Optional
import subprocess
import time
from cog import BasePredictor, Input, Path, BaseModel
import torch
from diffusers import (
    AutoPipelineForText2Image,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

SDXL_MODEL_URL = "https://weights.replicate.delivery/default/res-adapter/Lykon/dreamshaper-xl-1-0.tar"
SDXL_MODEL_WEIGHTS = "pretrained/Lykon/dreamshaper-xl-1-0"
SD15_MODEL_URL = "https://weights.replicate.delivery/default/res-adapter/dreamlike-art/dreamlike-diffusion-1.0.tar"
SD15_MODEL_WEIGHTS = "pretrained/dreamlike-art/dreamlike-diffusion-1.0"


class ModelOutput(BaseModel):
    without_res_adapter: Optional[Path]
    with_res_adapter: Path


def download_weights(url, dest, extract=True):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    args = ["pget"]
    if extract:
        args.append("-x")
    subprocess.check_call(args + [url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(SDXL_MODEL_WEIGHTS):
            download_weights(SDXL_MODEL_URL, SDXL_MODEL_WEIGHTS)
        if not os.path.exists(SD15_MODEL_WEIGHTS):
            download_weights(SD15_MODEL_URL, SD15_MODEL_WEIGHTS)

        # load "Lykon/dreamshaper-xl-1-0"
        self.sdxl_pipe = AutoPipelineForText2Image.from_pretrained(
            SDXL_MODEL_WEIGHTS, torch_dtype=torch.float16, variant="fp16"
        )
        self.sdxl_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.sdxl_pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++",
        )
        self.sdxl_pipe = self.sdxl_pipe.to("cuda")
   
        # load "dreamlike-art/dreamlike-diffusion-1.0"
        self.sd15_pipe = AutoPipelineForText2Image.from_pretrained(
            SD15_MODEL_WEIGHTS
        )  # fp16 not available for "dreamlike-art/dreamlike-diffusion-1.0"
        self.sd15_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.sd15_pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++",
        )
        self.sd15_pipe = self.sd15_pipe.to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        model_name: str = Input(
            description="Choose a stable diffusion model.",
            default="dreamlike-art/dreamlike-diffusion-1.0",
            choices=[
                "Lykon/dreamshaper-xl-1-0",
                "dreamlike-art/dreamlike-diffusion-1.0",
            ],
        ),
        resadapter_alpha: float = Input(
            description="Alpha for resadapter", ge=0, le=1, default=0.7,
        ),
        prompt: str = Input(
            description="Input prompt",
            default="Award-winning photo of a mystical fox girl fox in a serene forest clearing, sunlight filtering through the trees,ethereal,enchanting,vibrant orange fur,piercing amber eyes,delicate floral crown, flowing gown,surrounded by a gentle breeze, whispering leaves,magical atmosphere,captured by renowned photographer Emily Thompson using a Nikon D850,creating a dreamlike and captivating image",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="NSFW, poor bad amateur assignment cut out ugly",
        ),
        width: int = Input(description="Width of output image", default=1024),
        height: int = Input(description="Height of output image", default=1024),
        num_inference_steps: int = Input(
            description="Number of denoising steps", default=25
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=0, le=20, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        show_baseline: bool = Input(
            description="Show baseline without res-adapter for comparison.",
            default=True,
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)

        base_model = (
            "sd1.5" if model_name == "dreamlike-art/dreamlike-diffusion-1.0" else "sdxl"
        )

        if model_name == "Lykon/dreamshaper-xl-1-0":
            pipe = self.sdxl_pipe
        else:
            pipe = self.sd15_pipe

        if show_baseline:
            if len(pipe.get_active_adapters()) > 0:
                print("Unloading LoRA weights...")
                pipe.unload_lora_weights()

            print("Generating images without res_adapter...")
            baseline_image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                generator=generator,
                num_inference_steps=num_inference_steps,
            ).images[0]
            baseline_path = "/tmp/baseline.png"
            baseline_image.save(baseline_path)

        if len(pipe.get_active_adapters()) == 0:
            if base_model == "sd1.5":
                print("Loading Resolution LoRA weights...")
                pipe.load_lora_weights(
                    hf_hub_download(
                        repo_id="jiaxiangc/res-adapter",
                        subfolder="resadapter_v2_sd1.5",
                        filename="pytorch_lora_weights.safetensors",
                    ),
                    adapter_name="res_adapter",
                )
                print("Load Resolution Norm weights")
                pipe.unet.load_state_dict(
                    load_file(
                        hf_hub_download(
                            repo_id="jiaxiangc/res-adapter",
                            subfolder="resadapter_v2_sd1.5",
                            filename="diffusion_pytorch_model.safetensors",
                        )
                    ),
                    strict=False,
                )
            elif base_model == "sdxl":
                print("Loading Resolution LoRA weights...")
                pipe.load_lora_weights(
                    hf_hub_download(
                        repo_id="jiaxiangc/res-adapter",
                        subfolder="resadapter_v2_sdxl",
                        filename="pytorch_lora_weights.safetensors",
                    ),
                    adapter_name="res_adapter",
                )
                print("Load Resolution Norm weights")
                pipe.unet.load_state_dict(
                    load_file(
                        hf_hub_download(
                            repo_id="jiaxiangc/res-adapter",
                            subfolder="resadapter_v2_sdxl",
                            filename="diffusion_pytorch_model.safetensors",
                        )
                    ),
                    strict=False,
                )
        pipe.set_adapters(["res_adapter"], adapter_weights=[resadapter_alpha])

        print("Generating images with res_adapter...")
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        ).images[0]

        out_path = "/tmp/output.png"
        image.save(out_path)
        return ModelOutput(
            without_res_adapter=Path(baseline_path) if show_baseline else None,
            with_res_adapter=Path(out_path),
        )
