#!/bin/bash
mkdir -p ./models
cd models

wget -O dreamsim_checkpoint.zip https://github.com/ssundaram21/dreamsim/releases/download/v0.1.0/dreamsim_checkpoint.zip
wget -O clip_vitb32_pretrain.pth.tar https://github.com/ssundaram21/dreamsim/releases/download/v0.1.0/clip_vitb32_pretrain.pth.tar
wget -O clipl14_as_dino_vitl.pth.tar https://github.com/ssundaram21/dreamsim/releases/download/v0.1.0/clip_vitl14_pretrain.pth.tar
wget -O open_clip_vitb32_pretrain.pth.tar https://github.com/ssundaram21/dreamsim/releases/download/v0.1.0/open_clip_vitb32_pretrain.pth.tar
wget -O open_clipl14_as_dino_vitl.pth.tar https://github.com/ssundaram21/dreamsim/releases/download/v0.1.0/open_clip_vitl14_pretrain.pth.tar
wget -O mae_vib16_pretrain.pth https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
wget -O mae_vitl16_pretrain.pth https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth
wget -O mae_vith14_pretrain.pth https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth