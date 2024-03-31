# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from PIL import ImageDraw, ImageFont

import torch
from safetensors import safe_open
from torchvision import transforms
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    AutoPipelineForInpainting,
    AutoPipelineForImage2Image,
    StableDiffusionXLAdapterPipeline,
    AutoPipelineForText2Image,
    StableDiffusionAdapterPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
)
from diffusers.models import ControlNetModel
from diffusers.schedulers import (
    DPMSolverMultistepScheduler,
    LCMScheduler,
    UniPCMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
)


def load_text2image_pipeline(config):
    if config.diffusion_model.endswith(
        ".safetensors"
    ) or config.diffusion_model.endswith(".ckpt"):
        print(f"Load pipeline from civitai: {config.diffusion_model}")
        if config.model_type == "sd1.5":
            pipeline = StableDiffusionPipeline.from_single_file(
                config.diffusion_model,
                torch_dtype=torch.float16,
                variant="fp16",
                load_safety_checker=False,
                requires_safety_checker=False,
            )
        else:
            pipeline = StableDiffusionXLPipeline.from_single_file(
                config.diffusion_model,
                torch_dtype=torch.float16,
                variant="fp16",
                load_safety_checker=False,
                requires_safety_checker=False,
            )
    else:
        print(f"Load pipeline from huggingface: {config.diffusion_model}")
        if config.model_type == "sd1.5":
            pipeline = StableDiffusionPipeline.from_pretrained(
                config.diffusion_model,
                torch_dtype=torch.float16,
                variant="fp16",
                load_safety_checker=False,
                requires_safety_checker=False,
            )
        else:
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                config.diffusion_model,
                torch_dtype=torch.float16,
                variant="fp16",
                load_safety_checker=False,
                requires_safety_checker=False,
            )

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++",
    )

    # if config.timestep_spacing == "trailing":
    #     print("Detect timestep_spacing == trailing")
    #     pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")

    return pipeline


def load_text2image_lcm_lora_pipeline(config):
    if config.diffusion_model.endswith(
        ".safetensors"
    ) or config.diffusion_model.endswith(".ckpt"):
        print(f"Load pipeline from civitai: {config.diffusion_model}")
        if config.model_type == "sd1.5":
            pipeline = StableDiffusionPipeline.from_single_file(
                config.diffusion_model,
                torch_dtype=torch.float16,
                variant="fp16",
                load_safety_checker=False,
                requires_safety_checker=False,
            )
        else:
            pipeline = StableDiffusionXLPipeline.from_single_file(
                config.diffusion_model,
                torch_dtype=torch.float16,
                variant="fp16",
                load_safety_checker=False,
                requires_safety_checker=False,
            )
    else:
        print(f"Load pipeline from huggingface: {config.diffusion_model}")
        if config.model_type == "sd1.5":
            pipeline = StableDiffusionPipeline.from_pretrained(
                config.diffusion_model,
                torch_dtype=torch.float16,
                variant="fp16",
                load_safety_checker=False,
                requires_safety_checker=False,
            )
        else:
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                config.diffusion_model,
                torch_dtype=torch.float16,
                variant="fp16",
                load_safety_checker=False,
                requires_safety_checker=False,
            )

    pipeline.load_lora_weights(config.lcm_lora_path, adapter_name="lcm_lora")
    print(f"Load lcm-lora from {config.lcm_lora_path}")
    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)

    return pipeline


def load_controlnet_pipeline(config):
    controlnet = ControlNetModel.from_pretrained(
        config.controlnet_model, torch_dtype=torch.float16
    )

    if config.model_type == "sd1.5":
        if config.sub_task == "image_to_image":
            if config.diffusion_model.endswith(".safetensors") or config.diffusion_model.endswith(".ckpt"):
                pipeline = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
                    config.diffusion_model,
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                )
            else:
                pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                    config.diffusion_model,
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                )
        if config.sub_task == "text_to_image":
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                config.diffusion_model,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
    elif config.model_type == "sdxl":
        if config.sub_task == "image_to_image":
            if config.diffusion_model.endswith(".safetensors") or config.diffusion_model.endswith(".ckpt"):
                pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
                    config.diffusion_model,
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                )
            else:
                pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                    config.diffusion_model,
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                )

        if config.sub_task == "text_to_image":
            pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                config.diffusion_model,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    print(f"Load controlnet form {config.controlnet_model}")
    print(f"Load model form {config.diffusion_model}")

    return pipeline


def load_ip_adapter_pipeline(config):
    if config.sub_task == "image_variation":
        pipeline = AutoPipelineForText2Image.from_pretrained(
            config.diffusion_model, torch_dtype=torch.float16, safety_checker=None,
        )
    if config.sub_task == "image_to_image":
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            config.diffusion_model, torch_dtype=torch.float16, safety_checker=None,
        )
    if config.sub_task == "inpaint":
        pipeline = AutoPipelineForInpainting.from_pretrained(
            config.diffusion_model, torch_dtype=torch.float16, safety_checker=None,
        )
    if config.model_type == "sd1.5":
        if config.ip_adapter_weight_name == "general":
            WEIGHT_NAME = "ip-adapter_sd15.bin"
        elif config.ip_adapter_weight_name == "face":
            WEIGHT_NAME = "ip-adapter-full-face_sd15.bin"
        pipeline.load_ip_adapter(
            config.ip_adapter_model, subfolder="models", weight_name=WEIGHT_NAME
        )
    elif config.model_type == "sdxl":
        pipeline.load_ip_adapter(config.ip_adapter_model, subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

    pipeline.set_ip_adapter_scale(config.ip_adapter_scale)

    if config.ip_adapter_weight_name == "general":
        pipeline.scheduler = DDIMScheduler.from_config(
            pipeline.scheduler.config,
        )
    else:
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++",
        )
    

    print(f"Load from ip-adapter {config.ip_adapter_model}")
    return pipeline
