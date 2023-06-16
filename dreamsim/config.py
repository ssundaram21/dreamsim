dreamsim_args = {
        "model_config": {
            "feat_type": 'cls,embedding,embedding',
            "model_type": "dino_vitb16,clip_vitb16,open_clip_vitb16",
            "stride": "16,16,16",
            "lora": True
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

dreamsim_weights = "https://github.com/ssundaram21/dreamsim/releases/download/v0.1.0/dreamsim_checkpoint.zip"