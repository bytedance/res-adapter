# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
from typing import Optional
import subprocess
import time
from cog import BasePredictor, Input, Path, BaseModel
import torch
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler, EulerDiscreteScheduler, UNet2DConditionModel, StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

MODEL_URL = "https://weights.replicate.delivery/default/res-adapter/Lykon/dreamshaper-xl-1-0.tar"
MODEL_WEIGHTS = "pretrained/Lykon/dreamshaper-xl-1-0"

# For SDXL, SDXL-Lightning, dreamshaper-xl-1-0,
# For SDv1.5, dreamlike-diffusion-1.0

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
        if not os.path.exists(MODEL_WEIGHTS):
            download_weights(MODEL_URL, MODEL_WEIGHTS)
        self.default_pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_WEIGHTS, torch_dtype=torch.float16, variant="fp16"
        )
        self.default_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.default_pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++",
        )
        self.default_pipe = self.default_pipe.to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        base_model: str = Input(
            description="Choose a stable diffusion architecture, supporint sd1.5 and sdxl.",
            default="sdxl",
            choices=["sd1.5", "sdxl"],
        ),
        model_name: str = Input(
            description="Name of a stable diffusion model, should have either sd1.5 or sdxl architecture.",
            default="ByteDance/SDXL-Lightning",
            choice=["Lykon/dreamshaper-xl-1-0", "ByteDance/SDXL-Lightning", "dreamlike-art/dreamlike-diffusion-1.0"]
        ),
        prompt: str = Input(
            description="Input prompt",
            default="cinematic film still, photo of a girl, cyberpunk, neonpunk, headset, city at night, sony fe 12-24mm f/2.8 gm, close up, 32k uhd, wallpaper, analog film grain, SONY headset",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="ugly, deformed, noisy, blurry, nsfw, low contrast, text, BadDream, 3d, cgi, render, fake, anime, open mouth, big forehead, long neck",
        ),
        width: int = Input(
            description="Width of output image",
            default=512,
        ),
        height: int = Input(
            description="Height of output image",
            default=512,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", default=4
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=0, le=20, default=0
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

        if model_name == "Lykon/dreamshaper-xl-1-0":
            self.pipe = self.default_pipe

        elif model_name == "ByteDance/SDXL-Lightning":
            self.pipe = self.default_pipe
            repo = "ByteDance/SDXL-Lightning"
            ckpt = "sdxl_lightning_4step_unet.safetensors"  # Use the correct ckpt for your step setting!

            # Load SDXL-Lightning to UNet
            unet = self.default_pipe.unet
            unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))

            # Change UNet to pipeline
            self.pipe.unet = unet
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
        else:
            try:
                self.pipe = AutoPipelineForText2Image.from_pretrained(
                    model_name, torch_dtype=torch.float16, variant="fp16"
                )
            except:
                print("fp16 not available.")
                self.pipe = AutoPipelineForText2Image.from_pretrained(model_name)

            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="sde-dpmsolver++",
            )
            self.pipe = self.pipe.to("cuda")

        if show_baseline:
            if len(self.pipe.get_active_adapters()) > 0:
                print("Unloading LoRA weights...")
                self.pipe.unload_lora_weights()

            print("Generating images without res_adapter...")
            baseline_image = self.pipe(
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

        if len(self.pipe.get_active_adapters()) == 0:
            if base_model == "sd1.5":
                print("Loading Resolution LoRA weights...")
                self.pipe.load_lora_weights(
                    hf_hub_download(
                        repo_id="jiaxiangc/res-adapter",
                        subfolder=f"sd1.5",
                        filename="resolution_lora.safetensors",
                    ),
                    adapter_name="res_adapter",
                )
                self.pipe.set_adapters(["res_adapter"], adapter_weights=[1.0])
                print("Load Resolution Norm weights")
                self.pipe.unet.load_state_dict(load_file(
                    hf_hub_download(
                        repo_id="jiaxiangc/res-adapter",
                        subfolder="sd1.5",
                        filename="resolution_normalization.safetensors"
                    ),
                ), strict=False)
            elif base_model == "sdxl":
                print("Loading Resolution LoRA weights...")
                self.pipe.load_lora_weights(
                    hf_hub_download(
                        repo_id="jiaxiangc/res-adapter",
                        subfolder=f"sdxl-i",
                        filename="resolution_lora.safetensors",
                    ),
                    adapter_name="res_adapter",
                )
                self.pipe.set_adapters(["res_adapter"], adapter_weights=[1.0])

        print("Generating images with res_adapter...")
        image = self.pipe(
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
