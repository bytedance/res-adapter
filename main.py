# Inference for res-adapter with personalied diffusion models.
import argparse
import datetime
import os
from pathlib import Path
from PIL import Image

import cv2
import torch
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm
from omegaconf import OmegaConf

from resadapter.utils import (
    draw_text_on_images, 
    load_resadapter, 
    load_controlnet_pipeline, 
    load_ip_adapter_pipeline, 
    # load_t2i_adapter_pipeline, 
    load_text2image_pipeline,
    load_text2image_lcm_lora_pipeline,
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    return args

@torch.no_grad()
def main():
    args = parse_args()
    # #### 1. Get config ####
    config = OmegaConf.load(args.config)

    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if config.experiment_name != "":
        output_dir = f"samples/{config.experiment_name}-{time_str}"
    else:
        output_dir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(output_dir)
    print(f"Create {output_dir}")

    # #### 2.Load pipeline and scheduler ####
    if config.task == "t2i":
        pipeline = load_text2image_pipeline(config)
    if config.task == "t2i_accelerate":
        pipeline = load_text2image_lcm_lora_pipeline(config)
    if config.task == "controlnet":
        pipeline = load_controlnet_pipeline(config)
    # if config.task == "t2i_adapter":
        # pipeline = load_t2i_adapter_pipeline(config)    
    if config.task == "ip_adapter":
        pipeline = load_ip_adapter_pipeline(config)
        
    pipeline = pipeline.to(config.device)

    if config.enable_xformers:
        print("Enable xformers successfully.")
        pipeline.enable_xformers_memory_efficient_attention()

    # #### 3.Get prompts and other condition ####
    p_prompts = config.prompts
    n_prompt = config.n_prompt

    if config.task == "controlnet":
        condition_images = []
        source_images = []
        for image_path in config.source_images:
            source_image = Image.open(image_path)
            if config.scale_ratio:
                width, height = int(source_image.size[0]*config.scale_ratio), int(source_image.size[1]*config.scale_ratio)
            else:
                width, height = config.width, config.height
            source_image = source_image.resize((width, height))
            source_images.append(source_image)
            np_condition = np.array(source_image)
            np_condition = cv2.Canny(np_condition, 100, 200)
            np_condition = np_condition[:, :, None]
            np_condition = np.concatenate([np_condition, np_condition, np_condition], axis=2)
            condition_image = Image.fromarray(np_condition)
            condition_image.save(os.path.join(output_dir, f"condition_{Path(image_path).stem}.jpg"))
            condition_images.append(condition_image)

    if config.task == "t2i_adapter":
        condition_images = []
        for condition_path in config.condition_images:
            condition_image = Image.open(condition_path)
            if config.scale_ratio:
                width, height = int(condition_image.size[0]*config.scale_ratio), int(condition_image.size[1]*config.scale_ratio)
            else:
                width, height = config.width, config.height
            condition_image = condition_image.resize((width, height)).convert("L")
            condition_images.append(condition_image)

    if config.task == "ip_adapter":
        # Image Variation
        if config.sub_task == "image_variation":
            ip_adapter_images = []
            for ip_image_path in config.ip_adapter_images:
                ip_adpater_image = Image.open(ip_image_path)
                if config.scale_ratio:
                    width, height = int(ip_adpater_image.size[0]*config.scale_ratio), int(ip_adpater_image.size[1]*config.scale_ratio)
                else:
                    width, height = config.width, config.height

                ip_adpater_image = ip_adpater_image.resize((width, height))

                ip_adapter_images.append(ip_adpater_image)

        # Image to Image
        if config.sub_task == "image_to_image":
            ip_adapter_images = []
            source_images = []
            for ip_image_path, image_path in zip(config.ip_adapter_images, config.source_images):
                source_image = Image.open(image_path)
                if config.scale_ratio:
                    width, height = int(source_image.size[0]*config.scale_ratio), int(source_image.size[1]*config.scale_ratio)
                else:
                    width, height = config.width, config.height
                source_image = source_image.resize((width, height))
                source_images.append(source_image)

                ip_adapter_image = Image.open(ip_image_path)
                ip_adapter_image = ip_adapter_image.resize((width, height))
                ip_adapter_images.append(ip_adapter_image)
            
        # Image Inpainting
        if config.sub_task == "inpaint":
            source_images = []
            mask_images = []
            ip_adapter_images = []
            
            for ip_image_path, image_path, mask_path in zip(config.ip_adapter_images, config.source_images, config.mask_images):
                source_image = Image.open(image_path)
                if config.scale_ratio:
                    width, height = int(source_image.size[0]*config.scale_ratio), int(source_image.size[1]*config.scale_ratio)
                else:
                    width, height = config.width, config.height
                source_image = source_image.resize((width, height))
                source_images.append(source_image)

                mask_image = Image.open(mask_path)
                mask_image = mask_image.resize((width, height))
                mask_images.append(mask_image)

                ip_adapter_image = Image.open(ip_image_path)
                ip_adapter_image = ip_adapter_image.resize((width, height))
                ip_adapter_images.append(ip_adapter_image)


    # #### 4.Inference pipeline ####

    if config.seed:
        generator = torch.Generator(device=config.device).manual_seed(config.seed)
    else:
        generator = None

    if config.res_adapter_model == "":
        enable_compare = False
    else:
        enable_compare = config.enable_compare

    if enable_compare:
        # Inference baseline
        original_images = []
        for i, prompt in tqdm(enumerate(p_prompts), total=len(p_prompts), desc="[Baselines]: "):
            if config.task == "t2i" or config.task == "t2i_accelerate":
                kwargs = {}
            if config.task == "controlnet":
                if config.sub_task == "text_to_image":
                    kwargs = {"image": condition_images[i]}
                if config.sub_task == "image_to_image":
                    kwargs = {"control_image": condition_images[i], "image": source_images[i]}
            if config.task == "t2i_adapter":
                kwargs = {"image": condition_images[i]}
            if config.task == "ip_adapter":
                if config.sub_task == "image_variation":
                    kwargs = {"ip_adapter_image": ip_adapter_images[i]}
                if config.sub_task == "image_to_image":
                    kwargs = {"image": source_images[i], "ip_adapter_image": ip_adapter_images[i], "strength": 0.6}
                if config.sub_task == "inpaint":
                    kwargs = {"image": source_images[i], "mask_image": mask_images[i], "ip_adapter_image": ip_adapter_images[i], "strength": 0.5}
            
            images = pipeline(
                prompt=prompt,
                height=config.height,
                width=config.width,
                negative_prompt=n_prompt,
                num_inference_steps=config.num_inference_steps,
                num_images_per_prompt=config.num_images_per_prompt,
                generator=generator,
                output_type="pt",
                guidance_scale=config.guidance_scale,
                **kwargs,
            ).images
            original_images.append(images)

    # Load res-adapter
    if config.res_adapter_model != "":
        pipeline = load_resadapter(pipeline, config)
        print(f"Load res-adapter from {config.res_adapter_model}")
        pipeline.set_adapters(["res_adapter"], adapter_weights=[config.res_adapter_alpha])
    
    if config.task == "t2i_accelerate":
        pipeline.set_adapters(["res_adapter", "lcm_lora"], adapter_weights=[config.res_adapter_alpha, config.lcm_lora_alpha])

    # Inference with res-adapter
    resadapter_images = []
    for i, prompt in tqdm(enumerate(p_prompts), total=len(p_prompts), desc="[ResAdapter]: "):
        if config.task == "t2i" or config.task == "t2i_accelerate":
            kwargs = {}
        if config.task == "controlnet":
            if config.sub_task == "text_to_image":
                kwargs = {"image": condition_images[i]}
            if config.sub_task == "image_to_image":
                kwargs = {"control_image": condition_images[i], "image": source_images[i]}
        if config.task == "t2i_adapter":
            kwargs = {"image": condition_images[i]}
        if config.task == "ip_adapter":
            if config.sub_task == "image_variation":
                kwargs = {"ip_adapter_image": ip_adapter_images[i]}
            if config.sub_task == "image_to_image":
                kwargs = {"image": source_images[i], "ip_adapter_image": ip_adapter_images[i], "strength": 0.6}
            if config.sub_task == "inpaint":
                kwargs = {"image": source_images[i], "mask_image": mask_images[i], "ip_adapter_image": ip_adapter_images[i], "strength": 0.5}

        images = pipeline(
            prompt=prompt,
            height=config.height,
            width=config.width,
            negative_prompt=n_prompt,
            num_inference_steps=config.num_inference_steps,
            num_images_per_prompt=config.num_images_per_prompt,
            generator=generator,
            output_type="pt",
            guidance_scale=config.guidance_scale,
            **kwargs,
        ).images
        resadapter_images.append(images)

        # Save images
        texts = ["ResAdapter", "Baseline"]
        if enable_compare:
            for j in range(config.num_images_per_prompt):
                compare_image = torch.stack([resadapter_images[i][j], original_images[i][j]])
                if config.draw_text:
                    for k in range(len(texts)):
                        compare_image[k] = draw_text_on_images(compare_image[k], texts[k])
                
                if config.split_images:
                    for q in range(len(texts)):
                        save_image(
                            compare_image[q], os.path.join(output_dir, f"{prompt[:100]}_{j}_{texts[q]}.jpg"), normalize=True, value_range=(0, 1), nrow=2,
                        )
                else:
                    save_image(
                        compare_image, os.path.join(output_dir, f"{prompt[:100]}_{j}.jpg"), normalize=True, value_range=(0, 1), nrow=2,
                    )
        else:
            compare_image = resadapter_images[i]
            for m in range(config.num_images_per_prompt):
                save_image(
                    compare_image[m], os.path.join(output_dir, f"{prompt[:100]}_{m}.jpg"), normalize=True, value_range=(0, 1), nrow=2,
                )
        print(f"Saving image to {os.path.join(output_dir, f'{prompt[:100]}.jpg')}")

if __name__ == "__main__":
    main()