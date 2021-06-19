import timm
import torch
import torch.nn as nn

def create_ViT(img_size=224, patch_size=16, num_classes=10):
    model=timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=num_classes, img_size=img_size)
    if patch_size!=16:
        for p in model.named_parameters():
            if p[0]=='patch_embed.proj.bias':
                biases=p[1]
            if p[0]=='patch_embed.proj.weight':
                weights=p[1]
        sampling_step=16//patch_size
        sampled_weights=weights[:,:,::sampling_step,::sampling_step]
        model.patch_embed.proj=nn.Conv2d(3, 768, (patch_size, patch_size), (patch_size, patch_size))
        model.patch_embed.proj.weight=nn.Parameter(sampled_weights)
        model.patch_embed.proj.bias=nn.Parameter(biases)
        model.patch_embed.num_patches=(img_size//patch_size)**2
        model.pos_embed=nn.Parameter(torch.zeros(1, model.patch_embed.num_patches+1, 768))
    return model
