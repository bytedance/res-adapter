import os
from PIL import ImageDraw, ImageFont

import torch
from safetensors import safe_open
from torchvision import transforms
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionControlNetImg2ImgPipeline, AutoPipelineForInpainting, AutoPipelineForImage2Image, StableDiffusionXLAdapterPipeline, AutoPipelineForText2Image, StableDiffusionAdapterPipeline, StableDiffusionControlNetPipeline
from diffusers.models import ControlNetModel
from diffusers.schedulers import DPMSolverMultistepScheduler, LCMScheduler, UniPCMultistepScheduler, EulerAncestralDiscreteScheduler


def load_text2image_pipeline(config):
    if config.personalized_model.endswith(".safetensors") or config.personalized_model.endswith(".ckpt"):
        print(f"Load pipeline from civitai: {config.personalized_model}")
        if config.model_type == "sd1.5":
            pipeline = StableDiffusionPipeline.from_single_file(
                config.personalized_model,
                torch_dtype=torch.float16,
                variant="fp16",
                load_safety_checker=False,
                requires_safety_checker=False,
            )
        else:
            pipeline = StableDiffusionXLPipeline.from_single_file(
                config.personalized_model,
                torch_dtype=torch.float16,
                variant="fp16",
                load_safety_checker=False,
                requires_safety_checker=False,
            )
    else:
        print(f"Load pipeline from huggingface: {config.personalized_model}")
        if config.model_type == "sd1.5":
            pipeline = StableDiffusionPipeline.from_pretrained(
                config.personalized_model,
                torch_dtype=torch.float16,
                variant="fp16",
                load_safety_checker=False,
                requires_safety_checker=False,
            )
        else:
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                config.personalized_model,
                torch_dtype=torch.float16,
                variant="fp16",
                load_safety_checker=False,
                requires_safety_checker=False,
            )
    
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    
    return pipeline


def load_text2image_lcm_lora_pipeline(config):
    if config.personalized_model.endswith(".safetensors") or config.personalized_model.endswith(".ckpt"):
        print(f"Load pipeline from civitai: {config.personalized_model}")
        if config.model_type == "sd1.5":
            pipeline = StableDiffusionPipeline.from_single_file(
                config.personalized_model,
                torch_dtype=torch.float16,
                variant="fp16",
                load_safety_checker=False,
                requires_safety_checker=False,
            )
        else:
            pipeline = StableDiffusionXLPipeline.from_single_file(
                config.personalized_model,
                torch_dtype=torch.float16,
                variant="fp16",
                load_safety_checker=False,
                requires_safety_checker=False,
            )
    else:
        print(f"Load pipeline from huggingface: {config.personalized_model}")
        if config.model_type == "sd1.5":
            pipeline = StableDiffusionPipeline.from_pretrained(
                config.personalized_model,
                torch_dtype=torch.float16,
                variant="fp16",
                load_safety_checker=False,
                requires_safety_checker=False,
            )
        else:
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                config.personalized_model,
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
    controlnet = ControlNetModel.from_pretrained(config.controlnet_model, torch_dtype=torch.float16)

    if config.sub_task == "image_to_image":
        if config.personalized_model.endswith(".safetensors") or config.personalized_model.endswith(".ckpt"):
            pipeline = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
                config.personalized_model, controlnet=controlnet, torch_dtype=torch.float16
            )
        else:
            pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                config.personalized_model, controlnet=controlnet, torch_dtype=torch.float16
            )
    if config.sub_task == "text_to_image":
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            config.personalized_model, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
        )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    print(f"Load controlnet form {config.controlnet_model}")
    print(f"Load model form {config.personalized_model}")

    return pipeline
    

# def load_t2i_adapter_pipeline(config):
#     t2i_adapter = T2IAdapter.from_pretrained(config.t2i_adapter_model, torch_dtype=torch.float16)
#     pipeline = StableDiffusionAdapterPipeline.from_pretrained(
#         config.personalized_model, adapter=t2i_adapter, torch_dtype=torch.float16
#     )
#     # pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
#     pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

#     print(f"Load t2_adapter form {config.t2i_adapter_model}")
#     print(f"Load model form {config.personalized_model}")

#     return pipeline

def load_ip_adapter_pipeline(config):
    if config.sub_task == "image_variation":
        pipeline = AutoPipelineForText2Image.from_pretrained(config.personalized_model, torch_dtype=torch.float16)
    if config.sub_task == "image_to_image":
        pipeline = AutoPipelineForImage2Image.from_pretrained(config.personalized_model, torch_dtype=torch.float16)
    if config.sub_task == "inpaint":
        pipeline = AutoPipelineForInpainting.from_pretrained(config.personalized_model, torch_dtype=torch.float16)

    pipeline.load_ip_adapter(config.ip_adapter_model, subfolder="models", weight_name="ip-adapter_sd15.bin")
    pipeline.set_ip_adapter_scale(config.ip_adapter_scale)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    print(f"Load from ip-adapter {config.ip_adapter_model}")
    return pipeline


def load_resadapter(pipeline, config):
    NORM_WEIGHTS_NAME = "resolution_normalization.safetensors"
    LORA_WEIGHTS_NAME = "resolution_lora.safetensors"
    # Load resolution normalization
    norm_state_dict = {}
    with safe_open(os.path.join(config.res_adapter_model, NORM_WEIGHTS_NAME), framework="pt", device="cpu") as f:
        for key in f.keys():
            norm_state_dict[key] = f.get_tensor(key)
    m, u = pipeline.unet.load_state_dict(norm_state_dict, strict=False)
    
    # Load resolution lora
    pipeline.load_lora_weights(os.path.join(config.res_adapter_model, LORA_WEIGHTS_NAME), adapter_name="res_adapter")

    return pipeline

def draw_text_on_images(image, text):
    pil_image = transforms.ToPILImage()(image)
    draw = ImageDraw.Draw(pil_image)
    font_size = int(0.08 * image.shape[-1])
    font = ImageFont.truetype("assets/Times-Newer-Roman-Bold-Italic.otf", size=font_size)
    draw.text((10, 10), text, fill=(3, 176, 80), font=font)
    image = transforms.ToTensor()(pil_image)
    return image