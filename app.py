import os

os.system("pip install -U peft")
import random

import gradio as gr
import numpy as np
import PIL.Image

# import spaces
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

DESCRIPTION = """
# Res-Adapter :Domain Consistent Resolution Adapter for Diffusion Models
ByteDance provide a demo of [ResAdapter](https://huggingface.co/jiaxiangc/res-adapter) with [SDXL-Lightning-Step4](https://huggingface.co/ByteDance/SDXL-Lightning) to expand resolution range from 1024-only to 256~1024.
"""
if not torch.cuda.is_available():
    DESCRIPTION += (
        "\n<h1>Running on CPU 🥶 This demo does not work on CPU.</a> instead</h1>"
    )

MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1024"))
MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD") == "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# base = "stabilityai/stable-diffusion-xl-base-1.0"
base = "/mnt/bn/automl-aigc/chengjiaxiang/models/diffusers/dreamshaper-xl-1-0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors"  # Use the correct ckpt for your step setting!


# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt)))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")


# Load resadapter
pipe.load_lora_weights(
    hf_hub_download(
        repo_id="jiaxiangc/res-adapter",
        subfolder="sdxl-i",
        filename="resolution_lora.safetensors",
    ),
    adapter_name="res_adapter",
)

pipe = pipe.to(device)


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


# @spaces.GPU(enable_queue=True)
def generate(
    prompt: str,
    negative_prompt: str = "",
    prompt_2: str = "",
    negative_prompt_2: str = "",
    use_negative_prompt: bool = False,
    use_prompt_2: bool = False,
    use_negative_prompt_2: bool = False,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale_base: float = 5.0,
    num_inference_steps_base: int = 4,
    progress=gr.Progress(track_tqdm=True),
) -> PIL.Image.Image:
    print(f'** Generating image for: "{prompt}" **')
    generator = torch.Generator().manual_seed(seed)

    if not use_negative_prompt:
        prompt_2 = None  # type: ignore
    if not use_negative_prompt_2:
        negative_prompt_2 = None  # type: ignore

    pipe.set_adapters(["res_adapter"], adapter_weights=[0.0])
    base_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        prompt_2=prompt_2,
        negative_prompt_2=negative_prompt_2,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps_base,
        guidance_scale=guidance_scale_base,

        output_type="pil",
        generator=generator,
    ).images[0]


    pipe.set_adapters(["res_adapter"], adapter_weights=[1.0])
    res_adapt = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        prompt_2=prompt_2,
        negative_prompt_2=negative_prompt_2,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps_base,
        guidance_scale=guidance_scale_base,
        output_type="pil",
        generator=generator,
    ).images[0]

    return [res_adapt, base_image]





examples = [
    "A girl smiling",
    "A boy smiling",
    "A realistic photograph of an astronaut in a jungle, cold color palette, detailed, 8k",

]

theme = gr.themes.Base(
    font=[
        gr.themes.GoogleFont("Libre Franklin"),
        gr.themes.GoogleFont("Public Sans"),
        "system-ui",
        "sans-serif",
    ],
)
with gr.Blocks(css="footer{display:none !important}", theme=theme) as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    with gr.Group():
        prompt = gr.Text(
            label="Prompt",
            show_label=False,
            max_lines=1,
            container=False,
            placeholder="Enter your prompt",
        )
        run_button = gr.Button("Generate")
    # result = gr.Gallery(label="Left is Base and Right is Lora"),
    with gr.Accordion("Advanced options", open=False):
        with gr.Row():
            use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=True)
            use_prompt_2 = gr.Checkbox(label="Use prompt 2", value=False)
            use_negative_prompt_2 = gr.Checkbox(label="Use negative prompt 2", value=False)
        negative_prompt = gr.Text(
            label="Negative prompt",
            max_lines=1,
            placeholder="blur, cartoon, bad, face, painting",
            visible=False,
        )
        prompt_2 = gr.Text(
            label="Prompt 2",
            max_lines=1,
            placeholder="Enter your prompt",
            visible=False,
        )
        negative_prompt_2 = gr.Text(
            label="Negative prompt 2",
            max_lines=1,
            placeholder="Enter a negative prompt",
            visible=False,
        )

        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=0,
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        with gr.Row():
            width = gr.Slider(
                label="Width",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=512,
            )
            height = gr.Slider(
                label="Height",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=512,
            )
        with gr.Row():
            guidance_scale_base = gr.Slider(
                label="Guidance scale for base",
                minimum=0,
                maximum=1,
                step=0.1,
                value=0,
            )
            num_inference_steps_base = gr.Slider(
                label="Number of inference steps for base",
                minimum=1,
                maximum=50,
                step=1,
                value=4,
            )
    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=None,
        fn=generate,
        cache_examples=CACHE_EXAMPLES,
    )

    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        queue=False,
        api_name=False,
    )
    use_prompt_2.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_prompt_2,
        outputs=prompt_2,
        queue=False,
        api_name=False,
    )
    use_negative_prompt_2.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt_2,
        outputs=negative_prompt_2,
        queue=False,
        api_name=False,
    )
    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            prompt_2.submit,
            negative_prompt_2.submit,
            run_button.click,
        ],
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            prompt_2,
            negative_prompt_2,
            use_negative_prompt,
            use_prompt_2,
            use_negative_prompt_2,
            seed,
            width,
            height,
            guidance_scale_base,
            num_inference_steps_base,
        ],
        outputs=gr.Gallery(label="Right is Base and Left is ResAdapt with SDXL-ByteDance"),
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue(max_size=20, api_open=False).launch(show_api=False)