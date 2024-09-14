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
        "img_size": 224
    }

dreamsim_weights = {
    "ensemble": "https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_checkpoint.zip",
    "dino_vitb16": "https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_dino_vitb16_checkpoint.zip",
    "clip_vitb32": "https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_clip_vitb32_checkpoint.zip",
    "open_clip_vitb32": "https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_open_clip_vitb32_checkpoint.zip"
}
