## Prepare Environment

```bash
# Step1: Enter to res-adapter directory
cd res-adapter

# Step2: Install dependency
pip install -r requirements.txt

# Step3: Download diffusion models, and make the directory structure as follows:
models
├── res_adapter
│   ├── ...
├── diffusion_models
│   ├── ...
├── controlnet
│   ├── ...
├── ip_adapter
│   ├── ...
└── lcm-lora
    └──  ...
```

## Download

You only download models that you need for specific tasks. Here is an example:
- For text to image, you should download resadapter and personalized diffusion model.
- For controlnet task, you should download resadapter, controlnet and base diffusion model.
- For ip-adapter task, you should download resadapter, controlnet and base diffusion model.
- For lcm-lora task, you should download resadapter, lcm-lora and personalized diffusion model.

### ResAdapter

|Models  | Parameters | Resolution Range | Ratio Range | Links |
| --- | --- |--- | --- | --- |
|resadapter_v1_sd1.5| 0.9M | 128 <= x <= 1024 | 0.25 <= r <= 4 | [Download](https://huggingface.co/jiaxiangc/res-adapter)|
|resadapter_v1_sd1.5_extrapolation| 0.9M | 512 <= x <= 1024 | 0.25 <= r <= 4  | [Download](https://huggingface.co/jiaxiangc/res-adapter)|
|resadapter_v1_sd1.5_interpolation| 0.8M | 128 <= x <= 512 | 0.25 <= r <= 4  | [Download](https://huggingface.co/jiaxiangc/res-adapter)|
|resadapter_v1_sdxl| 0.5M | 256 <= x <= 1536 | 0.25 <= r <= 4  | [Download](https://huggingface.co/jiaxiangc/res-adapter) |
|resadapter_v1_sdxl_extrapolation| 0.5M | 1024 <= x <= 1536 | 0.25 <= r <= 4  | [Download](https://huggingface.co/jiaxiangc/res-adapter) |
|resadapter_v1_sdxl_interpolation| 0.4M | 256 <= x <= 1024 | 0.25 <= r <= 4  | [Download](https://huggingface.co/jiaxiangc/res-adapter) |


### Diffusion Models

We provide some personalized models for sampling style images with ResAdapter.
More personalized models can be found in [CivitAI](https://civitai.com/).

|Models  | Structure Type |Domain Type |Links |
| --- | --- |--- |--- |
| **Base model**
|SDv1.5 | - | General |[Download](https://huggingface.co/runwayml/stable-diffusion-v1-5)|
|SDXL1.0 |- | General |[Download](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| **Personalized model**
|RealisticVision|SDv1.5 |Realism | [Download](https://civitai.com/models/4201/realistic-vision-v60-b1)
|Dreamlike| SDv1.5 | Fantasy | [Download](https://civitai.com/models/1274/dreamlike-diffusion-10)
|DreamshaperXL|SDXL |2.5D | [Download](https://civitai.com/models/112902/dreamshaper-xl)
|...


### ControlNet, IP-Adapter and LCM-LoRA

|Modules | Name | Type | Links |
| --- |--- | --- | --- |
|ControlNet| lllyasviel/sd-controlnet-canny |SD1.5 | [Download](https://huggingface.co/lllyasviel/sd-controlnet-canny)
|ControlNet| diffusers/controlnet-canny-sdxl-1.0 |SDXL | [Download](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0)
|IP-Adapter| h94/IP-Adapter | SD1.5/SDXL | [Download](https://huggingface.co/h94/IP-Adapter)
|LCM-LoRA| latent-consistency/lcm-lora-sdv1-5 |SD1.5 | [Download](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5)
|LCM-LoRA| latent-consistency/lcm-lora-sdxl | SDXL| [Download](https://huggingface.co/latent-consistency/lcm-lora-sdxl)

