<div align="center">

<h1> ResAdapter: Domain Consistent Resolution Adapter for Diffusion Models  </h1>

Jiaxiang Cheng, Pan Xie*, Xin Xia, Jiashi Li, Jie Wu, Yuxi Ren, Huixia Li, Xuefeng Xiao, Min Zheng, Lean Fu (*Corresponding author)

AutoML, ByteDance Inc.

⭐ If ResAdapter is helpful to your images or projects, please help star this repo. Thanks! 🤗


<a href='https://res-adapter.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a> 
<a href='https://arxiv.org/abs/2403.02084'><img src='https://img.shields.io/badge/ Paper-Arxiv-red'></a> 
<a href='https://huggingface.co/papers/2403.02084'><img src='https://img.shields.io/badge/ Paper-Huggingface-blue'></a> 
![GitHub Org's stars](https://img.shields.io/github/stars/bytedance%2Fres-adapter)

[![Replicate](https://img.shields.io/badge/Replicate-Gradio-green)](https://replicate.com/bytedance/res-adapter)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-red)](https://huggingface.co/spaces/ameerazam08/Res-Adapter-GPU-Demo)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-ResAdapter-blue)](https://github.com/jiaxiangc/ComfyUI-ResAdapter)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=bytedance.res-adapter) 

**We propose ResAdapter, a plug-and-play resolution adapter for enabling any diffusion model generate resolution-free images: no additional training, no additional inference and no style transfer.**

<img src="assets/misc/dreamlike1.png" width="49.9%"><img src="assets/misc/dreamlike2.png" width="50%">
Comparison examples between resadapter and [dreamlike-diffusion-1.0](https://civitai.com/models/1274/dreamlike-diffusion-10).

</div>


## Release
- `[2024/03/30]` 🔥 We release the [ComfyUI-ResAdapter node](https://github.com/jiaxiangc/ComfyUI-ResAdapter).
- `[2024/03/28]` 🔥 We release the [all resadapter weights](https://huggingface.co/jiaxiangc/res-adapter).
- `[2024/03/12]` 🔥 We release the [inference code](https://github.com/bytedance/res-adapter/blob/main/main.py) and [gradio demo](https://replicate.com/bytedance/res-adapter).
- `[2024/03/04]` 🔥 We release the [arxiv paper](https://arxiv.org/abs/2403.02084).


## Quicktour

We provide a standalone [example code](quicktour.py) to help you quickly use resadapter with diffusion models.

<div align=center>

<img src="assets/misc/dreamshaper_resadapter.png" width="100%">
<img src="assets/misc/dreamshaper_baseline.png" width="100%">

Comparison examples (640x384) between resadapter and [dreamshaper-xl-1.0](https://huggingface.co/Lykon/dreamshaper-xl-1-0). Top: with resadapter. Bottom: without resadapter.

</div>

```python
# pip install diffusers, transformers, accelerate, safetensors, huggingface_hub
import torch
from torchvision.utils import save_image
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

generator = torch.manual_seed(0)
prompt = "portrait photo of muscular bearded guy in a worn mech suit, light bokeh, intricate, steel metal, elegant, sharp focus, soft lighting, vibrant colors"
width, height = 640, 384

# Load baseline pipe
model_name = "lykon-models/dreamshaper-xl-1-0"
pipe = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")

# Inference baseline pipe
image = pipe(prompt, width=width, height=height, num_inference_steps=25, num_images_per_prompt=4, output_type="pt").images
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
image = pipe(prompt, width=width, height=height, num_inference_steps=25, num_images_per_prompt=4, output_type="pt").images
save_image(image, f"image_resadapter.png", normalize=True, padding=0)
```

## Download

### Models

We have released all resadapter weights, you can download resadapter models from [Huggingface](https://huggingface.co/jiaxiangc/res-adapter/tree/main). The following is our resadapter model card:

|Models  | Parameters | Resolution Range | Ratio Range | Links |
| --- | --- |--- | --- | --- |
|resadapter_v1_sd1.5| 0.9M | 128 <= x <= 1024 | 0.25 <= r <= 4 | [Download](https://huggingface.co/jiaxiangc/res-adapter/tree/main/resadapter_v1_sd1.5)|
|resadapter_v1_sd1.5_extrapolation| 0.9M | 512 <= x <= 1024 | 0.25 <= r <= 4  | [Download](https://huggingface.co/jiaxiangc/res-adapter/tree/main/resadapter_v1_sd1.5_extrapolation)|
|resadapter_v1_sd1.5_interpolation| 0.8M | 128 <= x <= 512 | 0.25 <= r <= 4  | [Download](https://huggingface.co/jiaxiangc/res-adapter/tree/main/resadapter_v1_sd1.5_interpolation)|
|resadapter_v1_sdxl| 0.5M | 256 <= x <= 1536 | 0.25 <= r <= 4  | [Download](https://huggingface.co/jiaxiangc/res-adapter/tree/main/resadapter_v1_sdxl) |
|resadapter_v1_sdxl_extrapolation| 0.5M | 1024 <= x <= 1536 | 0.25 <= r <= 4  | [Download](https://huggingface.co/jiaxiangc/res-adapter/tree/main/resadapter_v1_sdxl_extrapolation) |
|resadapter_v1_sdxl_interpolation| 0.4M | 256 <= x <= 1024 | 0.25 <= r <= 4  | [Download](https://huggingface.co/jiaxiangc/res-adapter/tree/main/resadapter_v1_sdxl_interpolation) |

Hint1: We update the resadapter name format according to [controlnet](https://github.com/lllyasviel/ControlNet-v1-1-nightly).

Hint2: If you want use resadapter with personalized diffusion models, you should download them from [CivitAI](https://civitai.com/).

Hint3: If you want use resadapter with ip-adapter, controlnet and lcm-lora, you should download them from [Huggingface](https://huggingface.co/jiaxiangc/res-adapter).

Hint4: Here is an [installation guidance](models/README.md) for preparing environment and downloading models.

## Inference

If you want generate images in our inference script, you should download [related models](models/README.md) and fill in [configs](configs). Then you can directly run this script.

```bash
python main.py --config /path/to/file
```

<!-- ### 0️⃣ Best Practice
😄 For better image generation, we provide two advice:
- For text to image tasks, please use personalized models instead of base models.
- For other tasks, use base models. -->

### ResAdapter with Personalized Models for Text to Image

<div align=center>

<img src="assets/misc/dreamshaper-1024/resadapter1.jpg" width="25%"><img src="assets/misc/dreamshaper-1024/resadapter2.jpg" width="25%"><img src="assets/misc/dreamshaper-1024/resadapter3.jpg" width="25%"><img src="assets/misc/dreamshaper-1024/resadapter4.jpg" width="25%">
<img src="assets/misc/dreamshaper-1024/baseline1.jpg" width="25%"><img src="assets/misc/dreamshaper-1024/baseline2.jpg" width="25%"><img src="assets/misc/dreamshaper-1024/baseline3.jpg" width="25%"><img src="assets/misc/dreamshaper-1024/baseline4.jpg" width="25%">

Comparison examples (960x1104) between resadapter and [dreamshaper-7](https://civitai.com/models/1274/dreamlike-diffusion-10). Top: with resadapter. Bottom: without resadapter.

</div>

### ResAdapter with ControlNet for Image to Image

<div align=center>

<img src="assets/misc/controlnet/condition_bird.jpg" width="20%"><img src="assets/misc/controlnet/bird_1_ResAdapter.jpg" width="20%"><img src="assets/misc/controlnet/bird_2_ResAdapter.jpg" width="20%"><img src="assets/misc/controlnet/bird_3_ResAdapter.jpg" width="20%"><img src="assets/misc/controlnet/bird_4_ResAdapter.jpg" width="20%">
<img src="assets/misc/controlnet/condition_bird.jpg" width="20%"><img src="assets/misc/controlnet/bird_1_Baseline.jpg" width="20%"><img src="assets/misc/controlnet/bird_5_Baseline.jpg" width="20%"><img src="assets/misc/controlnet/bird_3_Baseline.jpg" width="20%"><img src="assets/misc/controlnet/bird_4_Baseline.jpg" width="20%">

Comparison examples (840x1264) between resadapter and [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny). Top: with resadapter, bottom: without resadapter.

</div>

### ResAdapter with ControlNet-XL for Image to Image

<div align=center>

<img src="assets/misc/controlnet-xl/condition_man.jpg" width="20%"><img src="assets/misc/controlnet-xl/man_0_ResAdapter.jpg" width="20%"><img src="assets/misc/controlnet-xl/man_1_ResAdapter.jpg" width="20%"><img src="assets/misc/controlnet-xl/man_2_ResAdapter.jpg" width="20%"><img src="assets/misc/controlnet-xl/man_3_ResAdapter.jpg" width="20%">
<img src="assets/misc/controlnet-xl/condition_man.jpg" width="20%"><img src="assets/misc/controlnet-xl/man_0_Baseline.jpg" width="20%"><img src="assets/misc/controlnet-xl/man_1_Baseline.jpg" width="20%"><img src="assets/misc/controlnet-xl/man_2_Baseline.jpg" width="20%"><img src="assets/misc/controlnet-xl/man_3_Baseline.jpg" width="20%">

Comparison examples (336x504) between resadapter and [diffusers/controlnet-canny-sdxl-1.0](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0). Top: with resadapter, bottom: without resadapter.

</div>

### ResAdapter with IP-Adapter for Face Variance

<div align=center>
<img src="assets/ip_adapter/ai_face2.png" width="20%"><img src="assets/misc/ip-adapter/resadapter3.jpg" width="20%"><img src="assets/misc/ip-adapter/resadapter4.jpg" width="20%"><img src="assets/misc/ip-adapter/resadapter5.jpg" width="20%"><img src="assets/misc/ip-adapter/resadapter7.jpg" width="20%">
<img src="assets/ip_adapter/ai_face2.png" width="20%"><img src="assets/misc/ip-adapter/baseline3.jpg" width="20%"><img src="assets/misc/ip-adapter/baseline4.jpg" width="20%"><img src="assets/misc/ip-adapter/baseline5.jpg" width="20%"><img src="assets/misc/ip-adapter/baseline7.jpg" width="20%">

Comparison examples (864x1024) between resadapter and [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter). Top: with resadapter, bottom: without resadapter.


</div>


### ResAdapter with LCM-LoRA for Speeding up

<div align=center>

<img src="assets/misc/lcm-lora/resadapter5.jpg" width="20%"><img src="assets/misc/lcm-lora/resadapter3.jpg" width="20%"><img src="assets/misc/lcm-lora/resadapter2.jpg" width="20%"><img src="assets/misc/lcm-lora/resadapter4.jpg" width="20%"><img src="assets/misc/lcm-lora/resadapter1.jpg" width="20%">
<img src="assets/misc/lcm-lora/baseline5.jpg" width="20%"><img src="assets/misc/lcm-lora/baseline3.jpg" width="20%"><img src="assets/misc/lcm-lora/baseline2.jpg" width="20%"><img src="assets/misc/lcm-lora/baseline4.jpg" width="20%"><img src="assets/misc/lcm-lora/baseline1.jpg" width="20%">

Comparison examples (512x512) between resadapter and [dreamshaper-xl-1.0](https://huggingface.co/Lykon/dreamshaper-xl-1-0) with [lcm-sdxl-lora](https://huggingface.co/latent-consistency/lcm-lora-sdxl). Top: with resadapter, bottom: without resadapter.


</div>

## Local Gradio Demo

Run the following script:

```bash
python app.py
```

## Usage Tips

- We recommend users to use **interpolation** version to generate lower-resolution images.
- We recommend users to use **extrapolation** version to generate higher-resolution images.
- We recommend users to use `resadapter_v1_sd1.5` and `resadapter_v1_sdxl` for deploying resadapter to generate images with broader resolution.
- We strongly recommend that you use the prompt corresponding to the personalized model, which helps to enhance the quality of the image.

## Acknowledgements

- ResAdapter is developed by AutoML Team at ByteDance Inc, all copyright reserved.
- Thanks to the [HuggingFace](https://huggingface.co/) gradio team for their free GPU support!
- Thanks to the [IP-Adapter](), [ControlNet](), [LCM-LoRA]() for their nice work.

## 🌟 Star History
[![Star History Chart](https://api.star-history.com/svg?repos=bytedance/res-adapter&type=Date)](https://star-history.com/#bytedance/res-adapter&Date)

## ✈️ Citation
If you find ResAdapter useful for your research and applications, please cite us using this BibTeX:
```
@article{cheng2024resadapter,
  title={ResAdapter: Domain Consistent Resolution Adapter for Diffusion Models},
  author={Cheng, Jiaxiang and Xie, Pan and Xia, Xin and Li, Jiashi and Wu, Jie and Ren, Yuxi and Li, Huixia and Xiao, Xuefeng and Zheng, Min and Fu, Lean},
  booktitle={arXiv preprint arxiv:2403.02084},
  year={2024}
}
```
For any question, please feel free to contact us via chengjiaxiang@bytedance.com or xiepan.01@bytedance.com.