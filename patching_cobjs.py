from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import sys
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import random

import plotly.express as px
import pandas as pd
from fancy_einsum import einsum
from cobjs_data import Example, NShotPrompt
from torch.utils.data import DataLoader, Dataset

def imshow(tensor, renderer=None, midpoint=0, **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=midpoint, color_continuous_scale="RdBu", **kwargs).show(renderer)

def line(tensor, renderer=None, **kwargs):
    px.line(y=utils.to_numpy(tensor), **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

from patching_utils import (logits_to_ave_logit_diff, 
                                ObjectData, patch_head_vector_at_pos,
                                cache_activation_hook,
                                patch_full_residual_component,
                                path_patching)


def load_json(filename):
    with open(filename, 'r') as fp:
        return json.load(fp)
    
if __name__ == '__main__':

    #load config
    cfg_fname = sys.argv[1]
    cfg = load_json(cfg_fname)
    cfg_vals = cfg.values()
    cfg = Namespace(**cfg)
    print("Using config", cfg)

    torch.set_grad_enabled(False)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_name = cfg.model_name#'gpt2-medium'
    print("Loading Model")
    model = HookedTransformer.from_pretrained(
        model_name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device = device
    )
    print("Done loading...")
    full_data = load_json("data/good_data_42.json")
    #with open("data/good_data_42.json", 'r') as fp:
    #    full_data = json.load(fp)#no reason not to use load_json
    mindiff_data = load_json('data/good_mindiff_data_42.json')
    #with open("data/correct_good_data_42.json", 'r') as fp:#gpt2-medium_correct_42.json", 'r') as fp:
    #    full_data = json.load(fp)
    #with open("data/correct_invalid_data_42.json") as fp:
    #    mindiff_data = json.load(fp)
        
    fulld, fulllabs = [i[0] for i in full_data], [i[1] for i in full_data]
    mindiffd, mindifflabs = [i[0] for i in mindiff_data], [i[1] for i in mindiff_data]
    batch_size = 25
    full_loader = DataLoader(ObjectData(fulld, fulllabs), batch_size=batch_size, shuffle=False)
    mindiff_loader = DataLoader(ObjectData(mindiffd, mindifflabs) , batch_size=batch_size, shuffle=False)
    
    #def path_patching(model, receiver_nodes, source_tokens, patch_tokens, ans_tokens, component='z', position=-1, freeze_mlps=False, indirect_patch=False):
    
    receiver_nodes = [(r[0], int(r[1]) if r[1] is not None else None) for r in cfg.receiver_nodes]
    component = cfg.component
    position = cfg.position
    freeze_mlps = cfg.freeze_mlps
    indirect_patch= cfg.indirect_patch

    output = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)

    for (inp, inp_labs), (co_inp, co_labs) in zip(full_loader, mindiff_loader):#zip(clean_loader, corr_loader):
        print('normal input', inp[0], inp_labs[0], 'mindiff', co_inp[0], 'co_labs', co_labs[0])
        inp_lab_toks = model.to_tokens(inp_labs, prepend_bos=False).squeeze(-1)
        colab_toks = model.to_tokens(co_labs, prepend_bos=False).squeeze(-1)
        ans_tokens = torch.stack([torch.tensor((inp_lab_toks[i], colab_toks[i] )) for i in range(len(inp_lab_toks))]).to(device)
        source_toks, cor_toks = model.to_tokens(inp, prepend_bos=False), model.to_tokens(co_inp, prepend_bos=False)
        #def path_patching(model, receiver_nodes, source_tokens, patch_tokens, ans_tokens, component='z', position=-1, freeze_mlps=False, indirect_patch=False, truncate_to_max_layer=True)
        output+=path_patching(model, receiver_nodes, source_toks, cor_toks, ans_tokens, component, position, freeze_mlps, indirect_patch)
    output/=len(full_loader)
    output = -output*100 #for visualization
    print("OUTPUT", output)
    recv_str = '_'.join(['-'.join([str(si) for si in s if si is not None]) for s in receiver_nodes])
    print("Saving to", f'results/cobjs_path_patching/{cfg_fname.strip(".json") }.npy')
    np.save(f'results/cobjs_path_patching/{cfg_fname.strip(".json") }.npy', output.numpy())

