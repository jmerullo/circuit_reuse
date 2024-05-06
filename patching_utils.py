
from torch.utils.data import DataLoader, Dataset
from functools import partial
import torch
from rich.progress import track
from transformer_lens import utils

def logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=False):
    # Only the final logits are relevant for the answer
    final_logits = logits[:, -1, :]
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
    answer_logit_diff = answer_logits[:, 0] - answer_logits[:, 1]

    if per_prompt:
        return answer_logit_diff
    else:
        return answer_logit_diff.mean()



class ObjectData(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)



def patch_head_vector_at_pos(
    clean_head_vector,
    hook,
    head_index,
    pos_index,
    corrupted_cache):
    clean_head_vector[:, pos_index, head_index, :] = corrupted_cache[hook.name][:, pos_index, head_index, :]
    return clean_head_vector

def cache_activation_hook(
    activation,
    hook,
    my_cache={}):
    #print("HOOK NAME:", hook.name)
    my_cache[hook.name] = activation
    return activation

def patch_full_residual_component(
    corrupted_residual_component, #: Float[torch.Tensor, "batch pos d_model"],
    hook,
    pos_index,
    corrupted_cache):
    corrupted_residual_component[:, pos_index, :] = corrupted_cache[hook.name][:, pos_index, :]
    return corrupted_residual_component


def path_patching(model, receiver_nodes, source_tokens, patch_tokens, ans_tokens, component='z', position=-1, freeze_mlps=False, indirect_patch=False, truncate_to_max_layer=True):
    model.reset_hooks()
    print("Component:", component)
    original_logits, cache= model.run_with_cache(source_tokens)
    original_logit_diff = logits_to_ave_logit_diff(original_logits, ans_tokens)
    #print('logits shape', original_logits.shape, original_logits[:, -1].shape)
    label_tokens = ans_tokens[:, 0]
    original_label_logits = original_logits[:, -1][list(range(len(original_logits))), label_tokens]
    #print("label logits", original_label_logits.shape)
    
    corr_logits, corrupted_cache= model.run_with_cache(patch_tokens)
    corrupt_logit_diff = logits_to_ave_logit_diff(corr_logits, ans_tokens)
    print(corrupt_logit_diff, original_logit_diff, 'DFF')
    del corr_logits
    patched_head_pq_diff = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
    
    def add_hook_to_attn(attn_block, hook_fn):
        if component=='v':
            attn_block.hook_v.add_hook(hook_fn)
        elif component=='q':
            attn_block.hook_q.add_hook(hook_fn)
        elif component == 'k':
            attn_block.hook_k.add_hook(hook_fn)
        elif component == 'z':
            attn_block.hook_z.add_hook(hook_fn)
        else:
            raise Exception(f"Component must be q,k,v, or z. You passed {component}")
    
    max_layer = model.cfg.n_layers
    if truncate_to_max_layer:
        target_layers = [r[0] for r in receiver_nodes]
        for t in target_layers:
            if type(t)==int:
                max_layer = min(t, max_layer)
        if max_layer<model.cfg.n_layers:
            max_layer+=1 #because we want to go up to max layer inclusive
    
    for layer in track(list(range(max_layer))):
        for head_index in range(model.cfg.n_heads):
            
            model.reset_hooks()
            if (layer, head_index) in receiver_nodes:
                continue
            
            #adding this before lets us cache the values before overwriting them
            receiver_cache = {}
            for recv_layer, recv_head in receiver_nodes:
                cache_fn = partial(cache_activation_hook, my_cache=receiver_cache)
                if recv_head is None:
                    #print("HOOK", recv_layer)
                    model.add_hook(recv_layer, cache_fn)
                else:
                    add_hook_to_attn(model.blocks[recv_layer].attn, cache_fn)
                    
            #Add the hooks for the sender nodes layer, head_index
            hook_fn = partial(patch_head_vector_at_pos, head_index=head_index, pos_index=position, corrupted_cache=corrupted_cache)
            model.blocks[layer].attn.hook_z.add_hook(hook_fn)
            
            for freeze_layer in list(range(model.cfg.n_layers)):
                if freeze_mlps:
                    hook_fn = partial(patch_full_residual_component, pos_index=position, corrupted_cache=cache)
                    model.blocks[freeze_layer].hook_mlp_out.add_hook(hook_fn)
                for freeze_head in range(model.cfg.n_heads):
                    if freeze_layer == layer and freeze_head == head_index:
                        continue
                    hook_fn = partial(patch_head_vector_at_pos, head_index=freeze_head, pos_index=position, corrupted_cache=cache) 
                    model.blocks[freeze_layer].attn.hook_z.add_hook(hook_fn)

            #Run with the original tokens with the layer, head_index as a sender node
            interv_logits, interv_cache = model.run_with_cache(source_tokens)
            model.reset_hooks()

            #now patch back in the receiver nodes that are changed by the sender nodes
            fwd_hooks = []
            for recv_layer, recv_head in receiver_nodes:
                if recv_head is None:
                    #print("HOOK", recv_layer, receiver_cache)
                    hook_fn = partial(patch_full_residual_component, pos_index=position, corrupted_cache=receiver_cache)
                    fwd_hooks.append((recv_layer, hook_fn))
                else:
                    hook_fn = partial(patch_head_vector_at_pos, head_index=recv_head, pos_index=position, corrupted_cache=receiver_cache)
                    fwd_hooks.append((utils.get_act_name(component, int(recv_layer), component), hook_fn))
            patched_logits = model.run_with_hooks(
                source_tokens,
                fwd_hooks = fwd_hooks,
                return_type="logits"
            )
            #patched_label_logits = patched_logits[:, -1][list(range(batch_size)), label_tokens]#[label_tokens]
            #patched_wrong_logits = patched_logits[:, -1][list(range(batch_size)), ans_tokens[:, 1]]#
            #print(patched_label_logits.shape, 'shape')
            #patched_logit_diff = ((patched_label_logits-original_label_logits)/original_label_logits)*100
            #print(patched_logit_diff.shape)
            #patched_logit_diff = logits_to_ave_logit_diff(patched_logits, ans_tokens)
            patched_logit_diff = logits_to_ave_logit_diff(patched_logits, ans_tokens)
            #print("PATHCHD", patched_logit_diff)
            patched_logit_diff = (patched_logit_diff-original_logit_diff)/(corrupt_logit_diff-original_logit_diff)
            patched_head_pq_diff[layer, head_index] = patched_logit_diff.item()
            #normalize_patched_logit_diff(patched_logit_diff.item()) #normalize_patched_logit_diff(patched_logit_diff)

            del patched_logits
            del patched_logit_diff
            #del patched_label_logits
    return patched_head_pq_diff
