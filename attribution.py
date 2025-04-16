
import math

import numpy as np
import torch
from torchvision.transforms import Resize, InterpolationMode

from tqdm import tqdm, trange

def permutation_generator(num_features: int, num_samples: int):
    for _ in range(num_samples):
        yield torch.randperm(num_features).tolist()

def compute_effect(model, x, y_pred, batch_size, patch_size, p_masks, pi_generator, device, base=np.zeros(3)):
    pi_list = []
    effect_list = []
    
    N = patch_size[0]*patch_size[1]
    img_size = x.shape[2:]
    resize = Resize(img_size, interpolation=InterpolationMode.NEAREST)

    x_diff = (x - torch.Tensor(base)[:,None,None]).to(device)

    for pi in tqdm(pi_generator):
        effects = np.zeros((N,N))

        for i in range(math.ceil(N/batch_size)):
            start = i*batch_size
            end = min(start+batch_size, N)
            n_mask = end-start

            pi_np = np.array(pi)
            
            masks = np.zeros((n_mask, N))
            default_val = np.ones((n_mask, 3, N)) * base[:, None] 
            for i in range(n_mask):
                masks[i, pi_np[:(start+i)]] = 1
                default_val[i, :, pi_np[:(start+i)]] = 0
    
            masks = resize(torch.Tensor(masks).reshape(-1, patch_size[0], patch_size[1])).unsqueeze(1)
            default_val = resize(torch.Tensor(default_val).reshape(-1, 3, patch_size[0], patch_size[1]))

            x_ = ((masks * x) + default_val).to(device).clone().detach().requires_grad_(True)           
            y_ = model(x_)[:, y_pred] 

            total_y = y_.sum()
            total_y.backward()

            grad_ = x_.grad

            effect = (x_diff*grad_).cpu().numpy()
            effect = np.sum(effect, axis=1)
            effect = np.einsum('ijk,ljk->il', effect, p_masks)
            
            effects[start:end] = effect

        torch.cuda.empty_cache()
        
        pi_list.append(pi_np)
        effect_list.append(effects)
    
    return effect_list, pi_list

def aggregate_interactions(model, x, y_pred, batch_size, patch_size, p_masks, pi_generator, device, base=np.zeros(3)):
    effect_list, pi_list = compute_effect(model, x, y_pred, batch_size, patch_size, 
                                      p_masks, pi_generator, device, base)
    
    N = patch_size[0]*patch_size[1]
    n_samples = len(pi_list)
    
    shapley_attr = np.zeros(N)
    api_attr = np.zeros(N)

    for k in range(n_samples):
        effects = effect_list[k]
        pi_np = pi_list[k]

        interaction = effects[1:] - effects[:-1]
        
        for i in range(0, N-1):
            interaction[i, pi_np[:(i+1)]] = 0

        original  = np.sum(interaction, axis=0) + effects[0]
        pos = np.sum(np.maximum(interaction,0), axis=0)
        aggregated = pos + effects[0]

        shapley_attr += original/n_samples
        api_attr += aggregated/n_samples
        
    return shapley_attr, api_attr