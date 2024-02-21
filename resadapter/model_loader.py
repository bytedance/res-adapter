import os

from safetensors import safe_open

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