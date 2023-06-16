import os
import torch
from .vision_transformer import vit_base, vit_large, vit_huge
from transformers import ViTMAEModel


def load_mae_as_vit(model_type, load_dir="./models"):
    if model_type == 'mae_vitb16':
        temp_model = vit_base()
        model = ViTMAEModel.from_pretrained("facebook/vit-mae-base", cache_dir=load_dir)

        temp_model.cls_token.data = model.state_dict()['embeddings.cls_token']
        temp_model.pos_embed.data = model.state_dict()['embeddings.position_embeddings']
        temp_model.patch_embed.proj.weight.data = model.state_dict()['embeddings.patch_embeddings.projection.weight']
        temp_model.patch_embed.proj.bias.data = model.state_dict()['embeddings.patch_embeddings.projection.bias']

        for i in range(12):
            temp_model.blocks[i].norm1.weight.data = model.state_dict()[f'encoder.layer.{i}.layernorm_before.weight']
            temp_model.blocks[i].norm1.bias.data = model.state_dict()[f'encoder.layer.{i}.layernorm_before.bias']
            temp_model.blocks[i].attn.qkv.weight.data = torch.cat((model.state_dict()[f'encoder.layer.{i}.attention.attention.query.weight'],
                                                             model.state_dict()[f'encoder.layer.{i}.attention.attention.key.weight'],
                                                             model.state_dict()[f'encoder.layer.{i}.attention.attention.value.weight']), dim=0)
            temp_model.blocks[i].attn.qkv.bias.data = torch.cat((model.state_dict()[f'encoder.layer.{i}.attention.attention.query.bias'],
                                                           model.state_dict()[f'encoder.layer.{i}.attention.attention.key.bias'],
                                                           model.state_dict()[f'encoder.layer.{i}.attention.attention.value.bias']), dim=0)
            temp_model.blocks[i].attn.proj.weight.data = model.state_dict()[f'encoder.layer.{i}.attention.output.dense.weight']
            temp_model.blocks[i].attn.proj.bias.data = model.state_dict()[f'encoder.layer.{i}.attention.output.dense.bias']
            temp_model.blocks[i].norm2.weight.data = model.state_dict()[f'encoder.layer.{i}.layernorm_after.weight']
            temp_model.blocks[i].norm2.bias.data = model.state_dict()[f'encoder.layer.{i}.layernorm_after.bias']
            temp_model.blocks[i].mlp.fc1.weight.data = model.state_dict()[f'encoder.layer.{i}.intermediate.dense.weight']
            temp_model.blocks[i].mlp.fc1.bias.data = model.state_dict()[f'encoder.layer.{i}.intermediate.dense.bias']
            temp_model.blocks[i].mlp.fc2.weight.data = model.state_dict()[f'encoder.layer.{i}.output.dense.weight']
            temp_model.blocks[i].mlp.fc2.bias.data = model.state_dict()[f'encoder.layer.{i}.output.dense.bias']

        temp_model.norm.weight.data = model.state_dict()['layernorm.weight']
        temp_model.norm.bias.data = model.state_dict()['layernorm.bias']

    elif model_type == 'mae_vitl16':
        temp_model = vit_large()
        path = os.path.join(load_dir, 'mae_vitl16_pretrain.pth')
        sd = torch.load(path)['model']
        temp_model.load_state_dict(sd)

    elif model_type == 'mae_vith14':
        temp_model = vit_huge()
        path = os.path.join(load_dir, 'mae_vith14_pretrain.pth')
        sd = torch.load(path)['model']
        temp_model.load_state_dict(sd)

    else:
        raise ValueError(f'Model {model_type} not supported')

    return temp_model
