# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
from typing import Optional
import subprocess
import time
from cog import BasePredictor, Input, Path
import torch
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
from huggingface_hub import hf_hub_download


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

MODEL_URL = "https://weights.replicate.delivery/default/res-adapter/Lykon/dreamshaper-xl-1-0.tar"
MODEL_WEIGHTS = "pretrained/Lykon/dreamshaper-xl-1-0"


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
        self.default_pipe.load_lora_weights(
            hf_hub_download(
                repo_id="jiaxiangc/res-adapter",
                subfolder=f"{base_model}-i",
                filename="resolution_lora.safetensors",
            ),
            adapter_name="res_adapter",
        )
        self.default_pipe.set_adapters(["res_adapter"], adapter_weights=[1.0])

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
            default="Lykon/dreamshaper-xl-1-0",
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
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", default=25
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        show_baseline: bool = Input(
            description="Show baseline without res-adapter for comparison.",
            default=True,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)

        if model_name == "Lykon/dreamshaper-xl-1-0":
            pipe = self.default_pipe
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_name, torch_dtype=torch.float16, variant="fp16"
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="sde-dpmsolver++",
            )
            pipe = pipe.to("cuda")

            pipe.load_lora_weights(
                hf_hub_download(
                    repo_id="jiaxiangc/res-adapter",
                    subfolder=f"{base_model}-i",
                    filename="resolution_lora.safetensors",
                ),
                adapter_name="res_adapter",
            )
            pipe.set_adapters(["res_adapter"], adapter_weights=[1.0])

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
        return Path(out_path)
