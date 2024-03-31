import torch
from torchvision.utils import save_image
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler, StableDiffusionPipeline

generator = torch.manual_seed(0)
prompt = "portrait photo of muscular bearded guy in a worn mech suit, light bokeh, intricate, steel metal, elegant, sharp focus, soft lighting, vibrant colors"
n_prompt = "ugly, deformed, noisy, blurry, nsfw, low contrast, text, BadDream, 3d, cgi, render, fake, anime, open mouth, big forehead, long neck"
width, height = 640, 384

# Load baseline pipe
model_name = "lykon-models/dreamshaper-xl-1-0"
pipe = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")

# Inference baseline pipe
image = pipe(prompt, negative_prompt=n_prompt, width=width, height=height, num_inference_steps=25, num_images_per_prompt=4, output_type="pt").images
save_image(image, f"image_baseline.png", normalize=True, padding=0)

# Load resadapter for baseline
resadapter_model_name = "resadapter_v1_sdxl"
pipe.load_lora_weights(
    hf_hub_download(repo_id="jiaxiangc/res-adapter", subfolder=resadapter_model_name, filename="pytorch_lora_weights.safetensors"), 
    adapter_name="res_adapter",
    ) # load lora weights
pipe.set_adapters(["res_adapter"], adapter_weights=[1.0])
pipe.unet.load_state_dict(
    load_file(hf_hub_download(repo_id="jiaxiangc/res-adapter", subfolder=resadapter_model_name, filename="diffusion_pytorch_model.safetensors")),
    strict=False,
    ) # load norm weights

# Inference resadapter pipe
image = pipe(prompt, negative_prompt=n_prompt, width=width, height=height, num_inference_steps=25, num_images_per_prompt=4, output_type="pt").images
save_image(image, f"image_resadapter.png", normalize=True, padding=0)