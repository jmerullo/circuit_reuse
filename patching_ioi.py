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
from tqdm import tqdm as tqdm
import plotly.express as px
import pandas as pd
from fancy_einsum import einsum
from cobjs_data import Example, NShotPrompt
from torch.utils.data import DataLoader, Dataset
from easy_transformer.ioi_dataset import (
    IOIDataset,
)


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


class ObjectData(Dataset):
    def __init__(self, data, labels, subjects):
        self.data = data
        self.labels = labels
        self.subjects = subjects

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.subjects[idx]

    def __len__(self):
        return len(self.data)

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
    model = HookedTransformer.from_pretrained(
        model_name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device = device
    )
    
    N=1000
    batch_size = 25
    ioi_dataset = IOIDataset(
        prompt_type="mixed",
        N=N,
        tokenizer=model.tokenizer,
        prepend_bos=False,
    )  # TODO make this a seeded dataset

    print(f"Here are two of the prompts from the dataset: {ioi_dataset.sentences[:2]}")
    print(len(ioi_dataset))
    #print(ioi_dataset.ioi_prompts[0]["S"])

    #abc here is really just flipping to make the minimal difference more like the Cobjs task
    abc_dataset = (ioi_dataset.gen_flipped_prompts(("S2", "IO")))
    print(len(abc_dataset))

    #convert IOI dataset to our format
    def extract_label(sentence):
        toks = model.to_str_tokens(sentence, prepend_bos=False)
        label = toks[-1]
        return ''.join(toks[:-1]), label

    cleand = []
    cleanlabs = []
    clean_s_words = []
    corruptd = []
    corruptlabs =[]

    for i, sentence in enumerate(ioi_dataset.sentences):
        subject = ioi_dataset.ioi_prompts[i]["S"]
        clean_s_words.append(' '+subject)
        s, l = extract_label(sentence)
        cleand.append(s)
        cleanlabs.append(l)

    for i, sentence in enumerate(abc_dataset.sentences):
        s, l = extract_label(sentence)
        corruptd.append(s)
        corruptlabs.append(l)
    print(cleand[0], cleanlabs[0], clean_s_words[0],)
    corr_loader = DataLoader(ObjectData(corruptd, corruptlabs, clean_s_words) , batch_size=batch_size, shuffle=False)
    clean_loader = DataLoader(ObjectData(cleand, cleanlabs, clean_s_words) , batch_size=batch_size, shuffle=False)
    
    #def path_patching(model, receiver_nodes, source_tokens, patch_tokens, ans_tokens, component='z', position=-1, freeze_mlps=False, indirect_patch=False):
    
    receiver_nodes = [(r[0], int(r[1]) if r[1] is not None else None) for r in cfg.receiver_nodes]
    component = cfg.component
    position = cfg.position
    freeze_mlps = cfg.freeze_mlps
    indirect_patch= cfg.indirect_patch

    output = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
    print(len(clean_loader))

    for (inp, inp_labs, subjs), (co_inp, _, _) in zip(clean_loader, corr_loader):
        #print(co_inp)
        #co_inp_labs = corruptlabs[0]
        #print(inp, inp_labs, model.to_tokens(inp_labs, prepend_bos=False))
        inp_lab_toks = model.to_tokens(inp_labs, prepend_bos=False).squeeze(-1)
        inp_subj_toks = model.to_tokens(subjs, prepend_bos=False).squeeze(-1)
        ans_tokens= torch.stack([torch.tensor((inp_lab_toks[i], inp_subj_toks[i])) for i in range(len(inp_lab_toks))]).to(device)

        #model.to_tokens(inp)
        source_toks, cor_toks = model.to_tokens(inp, prepend_bos=False), model.to_tokens(co_inp, prepend_bos=False)
        output+=path_patching(model, receiver_nodes, source_toks, cor_toks, ans_tokens, component, position, freeze_mlps, indirect_patch)
    output /= len(clean_loader)
    output = -output*100
    print("OUTPUT", output)
    recv_str = '_'.join(['-'.join([str(si) for si in s if si is not None]) for s in receiver_nodes])
    np.save(f'results/ioi_path_patching/{cfg_fname.strip(".json") }.npy', output.numpy())

