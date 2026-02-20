#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   sensitivity.py
@Time    :   2024/12/28 16:26:26
@Author  :   Jianmin Liu 
@Version :   1.0
@Site    :   https://jianmin.cc
@Desc    :   Measure the sensitivity of each coloumn of lora A.
'''

import torch 
from tqdm import tqdm
import torch.nn as nn 
from .layerwrapper import WrappedGPT, WrappedQGPT
from .calibration_data import get_loaders 
from peft.tuners.lora import LoraLayer
from transformers import BertForSequenceClassification

def find_lora_layers(module, name='', filter_by=None):
    """
    Find LoRA layers and support filtering.

    Args:
        module (nn.Module): PyTorch model or submodule.
        name (str): Name of the current module.
        filter_by (callable, optional): Filter function, filter LoRA layers.

    Returns:
        dict: Filtered LoRA layers.
    """
    if isinstance(module, LoraLayer):
        if filter_by is None or filter_by(module):
            return {name: module}
        else:
            return {}

    res = {}
    for name1, child in module.named_children():
        res.update(find_lora_layers(
            child, name=name + '.' + name1 if name != '' else name1, filter_by=filter_by
        ))
    return res

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device, model_type=None):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    max_length = model.config.max_position_embeddings
    if model_type =='lora':
        if isinstance(model.base_model.model,BertForSequenceClassification):
            layers = model.base_model.model.bert.encoder.layer
        else:
            layers = model.base_model.model.model.layers
    else:
        layers = model.model.layers
    
    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module, max_length):
            super().__init__()
            self.module = module
            self.max_length = max_length
            self.attention_type = getattr(module, 'attention_type', 'full_attention')
        def forward(self, *args, **kwargs):
            inps[cache['i']] = args[0]
            cache['i'] += 1
            if 'attention_mask' in kwargs: # Bert does not
                cache['attention_mask'] = kwargs['attention_mask']
            else:
                input_shape = args[0].size()
                attention_mask = torch.ones((input_shape[0], self.max_length), dtype=torch.long, device=args[0].device)
                # attention_mask = torch.ones(input_shape, dtype=torch.long, device=args[0].device)
                cache['attention_mask'] = attention_mask
            if 'position_ids' in kwargs:
                cache['position_ids'] = kwargs['position_ids']
            else:
                input_shape = args[0].size()
                position_ids = torch.arange(input_shape[1], dtype=torch.long, device=args[0].device)
                position_ids = position_ids.unsqueeze(0).expand(input_shape[0], -1)  
                cache['position_ids'] = position_ids
            raise ValueError
    layers[0] = Catcher(layers[0],max_length)
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    
    position_embeddings = None
    if model_type == 'lora':
        if not isinstance(model.base_model.model, BertForSequenceClassification):
            if hasattr(model.base_model.model.model, 'rotary_emb'):
                with torch.no_grad():
                    position_embeddings = model.base_model.model.model.rotary_emb(
                        inps[0:1], position_ids[0:1]
                    )
    else:
        if hasattr(model.model, 'rotary_emb'):
            with torch.no_grad():
                position_embeddings = model.model.rotary_emb(
                    inps[0:1], position_ids[0:1]
                )

    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids, position_embeddings 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity




def evaluate_lora_importance(model, calibdation_set, tokenizer,task_type,actual_task, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    nsamples = 128 # calibration data sample nums

    print("loading calibdation data")
    dataloader, _ = get_loaders(task_type,actual_task,nsamples,seed=0,seqlen=model.seqlen,tokenizer=tokenizer,dataset=calibdation_set)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(model, dataloader, device,model_type='lora')

    if isinstance(model.base_model.model,BertForSequenceClassification):
            layers = model.base_model.model.bert.encoder.layer
    else:
        layers = model.base_model.model.model.layers
    importance_scores, element_importance = {}, {}
    with tqdm(total=len(layers), 
            desc="Processing layers",
            unit="layer",
            dynamic_ncols=True,  
            mininterval=0.3,  
            miniters=1 
    ) as pbar:
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_lora_layers(layer)  
            # print(subset)

            if f"model.base_model.model.model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = model.hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
                if position_embeddings is not None:
                    position_embeddings = (position_embeddings[0].to(dev), position_embeddings[1].to(dev))

            wrapped_layers = {}
            for name in subset:
                from peft.tuners.lora.layer import Linear as f16Linear
                from peft.tuners.lora.bnb import Linear8bitLt
                '''
                <class 'peft.tuners.lora.layer.Linear'>
                <class 'peft.tuners.lora.bnb.Linear8bitLt'>
                '''
                if isinstance(subset[name], f16Linear):
                    wrapped_layers[name] = WrappedGPT(subset[name])
                elif isinstance(subset[name],Linear8bitLt):
                    # print("+++++++++++++")
                    # print(type(subset[name]))
                    # print(dir(type(subset[name])))
                    # print("++++++++++++++")
                    wrapped_layers[name] = WrappedQGPT(subset[name])
                else:
                    pass

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(nsamples):
                with torch.no_grad():
                    if isinstance(model.base_model.model,BertForSequenceClassification):
                        _ = layer(
                            inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                        )
                    else:
                        layer_kwargs = {
                            'hidden_states': inps[j].unsqueeze(0),
                            'attention_mask': attention_mask,
                            'position_ids': position_ids,
                        }
                        if position_embeddings is not None:
                            layer_kwargs['position_embeddings'] = position_embeddings
                        _ = layer(**layer_kwargs)
            for h in handles:
                h.remove()

            for name in subset:
                pbar.set_postfix_str(f"Current: {i} | Params: {len(subset)}{name[:15]}")

                lora_a = subset[name].lora_A['default'].weight
                x_norm_2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

                W_metric = torch.abs((lora_a * x_norm_2).sum(dim=0)) 
                W_metric_element = torch.abs(lora_a * x_norm_2)  
                importance_scores[f"layer_{i}_{name}"] = W_metric.detach().cpu().numpy()
                element_importance[f"layer_{i}_{name}"] = W_metric_element.detach().cpu().numpy()  
            pbar.update(1)


    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    return importance_scores, element_importance # dict(dict(layer,ndarray(4096)))

    