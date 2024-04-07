import os
import copy
import random

import gradio as gr
import numpy as np
import PIL.Image
# import spaces
import torch
from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


# 1.Description
title = r"""
<h1 align="center">ResAdapter: Domain Consistent Resolution Adapter for Diffusion Models</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://github.com/bytedance/res-adapter' target='_blank'><b>ResAdapter: Domain Consistent Resolution Adapter for Diffusion Models</b></a>.<br>
We propose ResAdapter, a plug-and-play resolution adapter for enabling any diffusion model generate resolution-free images: no additional training, no additional inference and no style transfer.<br>
How to use:<br>
1. Choose a personalized diffusion model.
2. Choose a resadapter weights according to the model type (sd1.5 or sdxl).
3. Change generation resolution of images.
4. Enter a text prompt, as done in normal text-to-image models.
5. Click the <b>Submit</b> button to begin customization.
"""

article = r"""
---
**Citation**
<br>
If our work is helpful for your research or applications, please cite us via:
```bibtex
@article{cheng2024resadapter,
  title={ResAdapter: Domain Consistent Resolution Adapter for Diffusion Models},
  author={Cheng, Jiaxiang and Xie, Pan and Xia, Xin and Li, Jiashi and Wu, Jie and Ren, Yuxi and Li, Huixia and Xiao, Xuefeng and Zheng, Min and Fu, Lean},
  booktitle={arXiv preprint arxiv:2403.02084},
  year={2024}
}
```
**Contact**
<br>
For any question, please feel free to contact us via chengjiaxiang@bytedance.com or xiepan.01@bytedance.com.</b>
<br>
**Acknowledgements**
This template is powered from [InstantID](https://huggingface.co/spaces/InstantX/InstantID).
"""

tips = r"""
### Usage tips of ResAdapter
1. If you are not satisfied with interpolation images, try to increase the alpha of resadapter to 1.0.
2. If you are not satisfied with extrapolate images, try to choose the alpha of resadapter in 0.3 ~ 0.7.
3. If you find the images with style conflicts, try to decrease the alpha of resadapter.
4. If you find resadapter is not compatible with other accelerate lora, try to decrease the alpha of resadapter to 0.5 ~ 0.7.
"""

# 2.Global variable
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "2048"))
MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD") == "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 3.Default model name

default_model_name = "dreamlike-art/dreamlike-diffusion-1.0"
default_pipe = AutoPipelineForText2Image.from_pretrained(default_model_name, torch_dtype=torch.float16)
default_pipe.scheduler = DPMSolverMultistepScheduler.from_config(default_pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
default_pipe = default_pipe.to(device)

# 4. Prepare examples
examples = [
    [
        "dreamlike-art/dreamlike-diffusion-1.0",
        "resadapter_v2_sd1.5",
        0.7,
        "Award-winning photo of a mystical fox girl fox in a serene forest clearing, sunlight filtering through the trees,ethereal,enchanting,vibrant orange fur,piercing amber eyes,delicate floral crown, flowing gown,surrounded by a gentle breeze, whispering leaves,magical atmosphere,captured by renowned photographer Emily Thompson using a Nikon D850,creating a dreamlike and captivating image",
        "NSFW, poor bad amateur assignment cut out ugly",
        1024,
        1024,
    ],
    [
        "dreamlike-art/dreamlike-diffusion-1.0",
        "resadapter_v2_sd1.5",
        0.7,
        "Pictures of you, beautiful face, youthful appearance, ultra focus, face iluminated, face detailed, ultra focus, dreamlike images, pixel perfect precision, ultra realistic, vibrant, ultra focus, face ilumined, face detailed, 8k resolution, watercolor, detailed colors, ultra focus, 8k resolution, watercolor, razumov style. art by Carne Griffiths, Frank Frazetta, sf, intricate artwork masterpiece, ominous, golden ratio, in the oil painting style reminiscent of Konstantin Razumov's work, yet interspersed with the layered paper illusion effect characteristic of Eiko Ojala, Reimagined splashes of ink in the digital art style, evoking at once impressions of Alberto Seveso's signature pieces, model standing confidently at the center, trending on cgsociety, intricate, epic, trending on artstation, by artgerm, h. r. giger and beksinski, highly detailed, vibrant, production cinematic character render, ultra high quality model, sf, intricate artwork masterpiece, ominous, matte painting movie poster, golden ratio, trending on cgsociety, intricate, epic, trending on artstation, by artgerm, h. r. giger and beksinski, highly detailed, vibrant",
        "NSFW, poor bad amateur assignment cut out ugly",
        1024,
        1024,
    ],
    [
        "Lykon/dreamshaper-xl-1-0",
        "resadapter_v2_sdxl",
        1.0,
        "(masterpiece), (extremely intricate), (realistic), portrait of a girl, the most beautiful in the world, (medieval armor), metal reflections, upper body, outdoors, intense sunlight, far away castle, professional photograph of a stunning woman detailed, sharp focus, dramatic, award winning, cinematic lighting, octane render unreal engine, volumetrics dtx, (film grain, blurry background, blurry foreground, bokeh, depth of field, sunset, motion blur), chainmail",
        "ugly, deformed, noisy, blurry, low contrast, text, BadDream, 3d, cgi, render, fake, anime, open mouth, big forehead, long neck",
        384,
        768,
    ],
    [
        "Lykon/dreamshaper-xl-1-0",
        "resadapter_v2_sdxl",
        1.0,
        "masterpiece, best quality, 1girl, sci-fi armor with black and red colors, glowing elements, redhair",
        "ugly, deformed, noisy, blurry, low contrast, text, BadDream, 3d, cgi, render, fake, anime, open mouth, big forehead, long neck",
        384,
        768,
    ]
]

# 5. Themes
theme = gr.themes.Base(
    font=[
        gr.themes.GoogleFont("Libre Franklin"),
        gr.themes.GoogleFont("Public Sans"),
        "system-ui",
        "sans-serif",
    ],
)

def run_for_examples(model_name, resadapter_model_name, resadapter_alpha, prompt, negative_prompt, width, height):
    return generate(
        model_name,
        resadapter_model_name,
        resadapter_alpha,
        prompt,
        negative_prompt,
        width,
        height,
        guidance_scale = 7.5,
        num_inference_steps = 25,
        seed = 44,
    )

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    # random seed
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def load_resadapter_for_pipe(pipe, resadapter_model_name, resadapter_alpha):
    # load lora
    pipe.load_lora_weights(
        hf_hub_download(repo_id="jiaxiangc/res-adapter", subfolder=resadapter_model_name, filename="pytorch_lora_weights.safetensors"), 
        adapter_name="res_adapter",
    )
    pipe.set_adapters(["res_adapter"], adapter_weights=[resadapter_alpha])
    # load normalization
    pipe.unet.load_state_dict(
        load_file(hf_hub_download(repo_id="jiaxiangc/res-adapter", subfolder=resadapter_model_name, filename="diffusion_pytorch_model.safetensors")),
        strict=False,
    )

    return pipe


# @spaces.GPU(enable_queue=True)
def generate(
    model_name: str,
    resadapter_model_name: str,
    resadapter_alpha: float,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 25,
    seed: int = 0,
) -> PIL.Image.Image:
    global default_model_name, default_pipe, device

    print(f'Generating image from: {prompt}')
    generator = torch.Generator().manual_seed(seed)
    
    if model_name == default_model_name:
        pipe = copy.deepcopy(default_pipe)
        pipe = pipe.to(device)

    else:
        pipe = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
        pipe = pipe.to(device)
        default_pipe = copy.deepcopy(pipe)
        default_model_name = model_name

    # inference baseline
    base_image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        output_type="pil",
        generator=generator,
    ).images[0]

    # inference resadapter
    pipe = load_resadapter_for_pipe(pipe, resadapter_model_name, resadapter_alpha)
    resadapter_image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        output_type="pil",
        generator=generator,
    ).images[0]

    return [resadapter_image, base_image]


# 6. UI
with gr.Blocks(css="footer{display:none !important}", theme=theme) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                model_name_choices = [
                    "dreamlike-art/dreamlike-diffusion-1.0",
                    "Lykon/dreamshaper-xl-1-0",
                ]
                model_name = gr.Dropdown(
                    label="model name",
                    choices=model_name_choices,
                    value="dreamlike-art/dreamlike-diffusion-1.0",
                )
                resadapter_model_name_choices = ["resadapter_v2_sd1.5", "resadapter_v2_sdxl"]
                resadapter_model_name = gr.Dropdown(
                    label="resadapter model name",
                    choices=resadapter_model_name_choices,
                    value="resadapter_v2_sd1.5",
                )
                resadapter_alpha = gr.Slider(
                    label="resadapter alpha",
                    minimum=0,
                    maximum=1.0,
                    step=0.01,
                    value=0.7,
                )
            with gr.Column():
                prompt = gr.Text(
                    label="Prompt",
                    max_lines=1,
                    placeholder="Enter your prompt",
                    visible=True,
                )
                negative_prompt = gr.Text(
                    label="Negative prompt",
                    max_lines=1,
                    placeholder="NSFW, poor bad amateur assignment cut out ugly",
                    visible=True,
                )
                run_button = gr.Button("Submmit")
            width = gr.Slider(
                label="Width",
                minimum=128,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=1024,
            )
            height = gr.Slider(
                label="Height",
                minimum=128,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=1024,
            )
            guidance_scale = gr.Slider(
                label="CFG Scale",
                minimum=0,
                maximum=20,
                step=0.5,
                value=7.5,
            )
            num_inference_steps = gr.Slider(
                label="Sampling steps",
                minimum=1,
                maximum=50,
                step=1,
                value=25,
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
            resadapter_output = gr.Image(label="Resadapter images")
            baseline_output = gr.Image(label="Baseline images")
    
    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
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
            model_name,
            resadapter_model_name,
            resadapter_alpha,
            prompt,
            negative_prompt,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            seed,
        ],
        outputs=[resadapter_output, baseline_output],
        api_name="run",
    )

    gr.Examples(
        examples=examples,
        inputs=[model_name, resadapter_model_name, resadapter_alpha, prompt, negative_prompt, width, height],
        outputs=[resadapter_output, baseline_output],
        fn=run_for_examples,
        cache_examples="lazy",
    )
    gr.Markdown(tips)
    gr.Markdown(article)

if __name__ == "__main__":
    demo.queue(max_size=20, api_open=False).launch(show_api=False, server_port=5002)