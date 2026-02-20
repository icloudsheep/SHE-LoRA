#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   client_utils.py
@Time    :   2025/03/04 14:52:57
@Author  :   Jianmin Liu 
@Version :   1.0
@Site    :   https://jianmin.cc
@Desc    :   Tools for client training
'''
import pickle
from peft import (
    # LoraConfig,
    # get_peft_model,
    get_peft_model_state_dict,
    # set_peft_model_state_dict,
)
from flowertune_llm.models import (
    get_parameters,
)
from ..she.ckks_client import (
    exchange_columns,
    encrypt_cipher_list,
    decrypt
)
import tenseal as ts
import numpy as np
from flwr.common.logger import log
from logging import INFO
from ..ope import OPE

def set_enc_lines_to_client(enc_lines: dict,he_budget: int):
    """
    Cut the negotiated encryption range according to the budget and save it to the client's properties.
    """
    client_enc = {}
    ope = OPE()
    for layer,enc in enc_lines.items():
        # print("Layer: ",layer," EncLines: ",enc)
        client_enc[layer] = [ope.decrypt(enc_item) for enc_item in enc[:he_budget]]
    return client_enc


def handle_parameters_to_server(model,enc_lines:dict,he_budget:int):
    parameters = get_parameters(model)
    index_as,enc_a_lines = find_loraAid_and_enc_lines(model,enc_lines)
    plain,cipher = get_plain_cipher(index_as=index_as,enc_a_lines=enc_a_lines,parameters=parameters)
    cipher_bytes = encrypt_cipher_list(cipher)
    cipher_size_mb = len(cipher_bytes) / (1024 * 1024)  
    log(
        INFO,
        f"Ciphertext size: {cipher_size_mb:.2f} MB"
    )
    return plain, cipher_bytes


def find_loraAid_and_enc_lines(peftmodel,peft_enc_lines):
    state_dict = get_peft_model_state_dict(peftmodel)
    lora_indices = [i for i, name in enumerate(state_dict.keys()) if 'lora_A.weight' in name]
    enc_lines_a_layer = [enc for enc in peft_enc_lines.values()]
    return lora_indices,enc_lines_a_layer
    
def get_plain_cipher(index_as,enc_a_lines,parameters):
    assert len(index_as) == len(enc_a_lines), (
        f"Parameter matrix and encryption range number mismatch, please check the input encryption row number and parameter matrix."
        f" Current parameter matrix length: {len(index_as)}, current encryption row number: {len(enc_a_lines)}"
    )
    cipher_list = []
    for i,j in zip(index_as,range(len(enc_a_lines))):  
        current_layer_a = parameters[i]
        current_exchange_line = enc_a_lines[j]
        change_layer_a = exchange_columns(A=current_layer_a,selected_cols=current_exchange_line)
        columns_to_enc = change_layer_a[:,-len(current_exchange_line):].copy()  
        change_layer_a[:,-len(current_exchange_line):] = 0   
        parameters[i]= change_layer_a
        cipher_list.append(columns_to_enc)
    return parameters,cipher_list



def plain_adaptive_rank(parameters, rank):
    if parameters[-1].ndim ==1:
        last_weight_bais = parameters[-2:]
        parameters = parameters[:-2]

    sliced_parameters = []
    for i, param in enumerate(parameters):
        if i % 2 == 0:  
            sliced_param = param[:rank, :] 
        else:  
            sliced_param = param[:, :rank]          
        sliced_parameters.append(sliced_param)
    if parameters[-1].ndim ==1:
        parameters.extend(last_weight_bais)
    return sliced_parameters
    
    

def fusion_plain_cipher(plain_agg_results,cipher_agg_results,enc_a_lines,max_rank):
    parameters = []  
    cipher_to_plain = decode_ciphers(cipher_agg_results) 
    assert len(enc_a_lines)==len(cipher_to_plain),f"Cipher block number {len(enc_a_lines)} and Layer A number {len(cipher_agg_results)} mismatch"
    for index,layer_martix in enumerate(cipher_to_plain):
        U, s, Vh = np.linalg.svd(layer_martix.transpose(), full_matrices=False)
        A_e = Vh                               
        B_e = U@ np.diag(s)                    
        A_p = plain_agg_results[index*2]   
        B_p = plain_agg_results[index*2+1] 
        B_r = np.hstack([B_p, B_e])   
        A_r =  np.block([
                [A_p ,np.zeros((A_p.shape[0],A_e.shape[1]))], 
                [np.zeros((A_e.shape[0],A_p.shape[1]),dtype=float),A_e] 
            ])  # [3200,3204]
        A_r = exchange_columns(A_r,selected_cols=next(iter(enc_a_lines.values())) if enc_a_lines else None)
        result = B_r.dot(A_r)
        U, s, Vh = np.linalg.svd(result, full_matrices=False)
        B_aggregated = U[:, :max_rank] @ np.diag(s[:max_rank])  
        A_aggregated = Vh[:max_rank, :len(A_p[1])]  
        parameters.append(A_aggregated)
        parameters.append(B_aggregated)

    return parameters

def decode_ciphers(cipher_bytes):
    with open("./flowertune_llm/she/ckks_full_context.bytes", "rb") as f:
        context = ts.context_from(f.read())

    results = []
    cipher_layers = pickle.loads(cipher_bytes)  
    for cipher_list in cipher_layers: 
        layer_res = [] 
        for enc_line in cipher_list:            
            ckks_tensor = ts.ckks_vector_from(context,enc_line['par'])
            Weighted_average_coefficient = enc_line['num']
            ckks_plain = np.array(ckks_tensor.decrypt(),dtype="float32") 
            ckks_plain *= 1/Weighted_average_coefficient
            layer_res.append(ckks_plain)
        layer_martix = np.vstack(layer_res)
        results.append(layer_martix)
    return results
