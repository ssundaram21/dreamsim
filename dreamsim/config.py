dreamsim_args = {
        "model_config": {
            "ensemble": {
                "feat_type": 'cls,embedding,embedding',
                "model_type": "dino_vitb16,clip_vitb16,open_clip_vitb16",
                "stride": "16,16,16",
                "lora": True
            },
            "dino_vitb16": {
                "feat_type": 'cls',
                "model_type": "dino_vitb16",
                "stride": "16",
                "lora": True
            },
            "clip_vitb32": {
                "feat_type": 'embedding',
                "model_type": "clip_vitb32",
                "stride": "32",
                "lora": True
            },
            "open_clip_vitb32": {
                "feat_type": 'embedding',
                "model_type": "open_clip_vitb32",
                "stride": "32",
                "lora": True
            }
        },
        "lora_config": {
            "r": 16,
            "lora_alpha": 0.5,
            "lora_dropout": 0.3,
            "bias": "none",
            "target_modules": ['qkv']
        },
        "img_size": 224
    }

dreamsim_weights = {
    "ensemble": "https://github.com/ssundaram21/dreamsim/releases/download/v0.1.0/dreamsim_checkpoint.zip",
    "dino_vitb16": "https://github.com/ssundaram21/dreamsim/releases/download/v0.1.2/dreamsim_dino_vitb16_checkpoint.zip",
    "clip_vitb32": "https://github.com/ssundaram21/dreamsim/releases/download/v0.1.2/dreamsim_clip_vitb32_checkpoint.zip",
    "open_clip_vitb32": "https://github.com/ssundaram21/dreamsim/releases/download/v0.1.2/dreamsim_open_clip_vitb32_checkpoint.zip"
}
