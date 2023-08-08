from einops import rearrange,repeat,reduce
import torch
import torch.nn as nn

patch_size = 8
H = 32

def patchify(imgs):
    
    p = patch_size
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2))
    return x
    
def unpatchify(x):
    
    p = patch_size
    h = w = H
    assert h * w == x.shape[1]
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
    return imgs

def MAE_visualize(img, y, mask):
    
    p = 8
    h = w = 32
    y = unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, patch_size**2)  
    mask = unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    x = torch.einsum('nchw->nhwc', img).cpu()
    # masked image
    im_masked = x * (1 - mask)
    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask
    im_paste = rearrange(im_paste,'b w h c -> b c w h')
    im_masked = rearrange(im_masked,'b w h c -> b c w h')
    y = rearrange(y,'b w h c -> b c w h')
    return y[0],im_masked[0],im_paste[0]