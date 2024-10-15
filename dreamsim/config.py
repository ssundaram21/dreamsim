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
            "dinov2_vitb14": {
                "feat_type": 'cls',
                "model_type": "dinov2_vitb14",
                "stride": "14",
                "lora": True
            },
             "dino_vitb16_patch": {
                "feat_type": 'cls_patch',
                "model_type": "dino_vitb16",
                "stride": "16",
                "lora": True
            },
            "dinov2_vitb14_patch": {
                "feat_type": 'cls_patch',
                "model_type": "dinov2_vitb14",
                "stride": "14",
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
            },
            "synclr_vitb16": {
                "feat_type": 'cls',
                "model_type": "synclr_vitb16",
                "stride": "16",
                "lora": True
            },
        },
        "img_size": 224
    }

dreamsim_weights = {
    "ensemble": "https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_ensemble_checkpoint.zip",
    "dino_vitb16": "https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_dino_vitb16_checkpoint.zip",
    "clip_vitb32": "https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_clip_vitb32_checkpoint.zip",
    "open_clip_vitb32": "https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_open_clip_vitb32_checkpoint.zip",
    "dinov2_vitb14": "https://github.com/ssundaram21/dreamsim/releases/download/v0.2.1-checkpoints/dreamsim_dinov2_vitb14_checkpoint.zip",
    "synclr_vitb16": "https://github.com/ssundaram21/dreamsim/releases/download/v0.2.1-checkpoints/dreamsim_synclr_vitb16_checkpoint.zip",
    "dino_vitb16_patch": "https://github.com/ssundaram21/dreamsim/releases/download/v0.2.1-checkpoints/dreamsim_dino_vitb16_patch_checkpoint.zip",
    "dinov2_vitb14_patch": "https://github.com/ssundaram21/dreamsim/releases/download/v0.2.1-checkpoints/dreamsim_dinov2_vitb14_patch_checkpoint.zip"
}
