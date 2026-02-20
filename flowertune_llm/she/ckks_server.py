#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   utils_server.py
@Time    :   2024/12/27 15:18:43
@Author  :   Jianmin Liu 
@Version :   1.0
@Site    :   https://jianmin.cc
@Desc    :   Server side functions for SHE-LoRA.
'''
import numpy as np
import tenseal as ts
from tenseal.enc_context import Context
import pickle
from ..she import decrypt
from concurrent.futures import ProcessPoolExecutor
import os

def split_matrix(A_1, num_enc_col):
    """
    Split `A_1` into two matrices, the first matrix only contains the first `num_enc_col` columns, the last column is 0, and the second matrix only contains the last few columns. The sum of the two matrices equals `A_1`.
    """
    first_matrix = np.zeros_like(A_1)
    first_matrix[:, :num_enc_col] = A_1[:, :num_enc_col]
    second_matrix = np.zeros_like(A_1)
    second_matrix[:, num_enc_col:] = A_1[:, num_enc_col:]
    return first_matrix, second_matrix



def aggregate_ckks_plain_with_blocks(matrices, block_size=8):
    num_rows = matrices[0].shape[0]
    for mat in matrices:
        assert mat.shape[0] == num_rows, "All matrices must have the same number of rows"

    max_cols = max(mat.shape[1] for mat in matrices)
    result = np.zeros((num_rows, max_cols), dtype=np.float64)
    count_matrix = np.zeros((num_rows, max_cols), dtype=np.int64)

    # Process columns in blocks
    for start_col in range(0, max_cols, block_size):
        end_col = min(start_col + block_size, max_cols)

        # Process columns in blocks
        for mat in matrices:
            num_cols = mat.shape[1]
            if num_cols + start_col < max_cols:
                block_start = max(0, max_cols - num_cols - start_col)
                block_end = min(end_col - start_col, num_cols)

                if block_start < block_end:
                    aligned_start = max_cols - num_cols + block_start
                    aligned_end = aligned_start + block_end - block_start

                    result[:, start_col:end_col] += mat[:, block_start:block_end]
                    count_matrix[:, start_col:end_col] += 1

    # Calculate weighted average (avoid division by 0)
    np.divide(result, count_matrix, out=result, where=(count_matrix != 0))

    return result

def aggregate_ckks_tensors(item_list, context=None):
    """
    Aggregate CKKS tensors for a layer.
    Take the last CKKS tensor for aggregation, and add the result to the result list in reverse order.
    Remove the last element from the item after each aggregation.
    """
    history_results = [] 
    current_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_path, "ckks_full_context.bytes"), "rb") as f:
        context = ts.context_from(f.read())
    while True:
        if all(not item for item in item_list):
            break
        result = None
        num_tensors = 0
        line_res = {}
        for item in item_list:  
            if not item:
                continue
            ckks_tensor = item.pop()
            if isinstance(ckks_tensor,bytes):
                ckks_tensor = ts.ckks_vector_from(context,ckks_tensor)
            if result is None:
                result = ckks_tensor
            else:
                result += ckks_tensor

            num_tensors += 1

        if result is not None and num_tensors > 0:
            line_res['par'] = result.serialize()
            line_res['num'] = num_tensors
            history_results.insert(0, line_res)
    return history_results

def analyze_client_data(client_data_list):
    client_data_list = sorted(client_data_list, key=lambda x: x['extent'])
    client_data_list = sorted(client_data_list, key=lambda x: x['rank'])
    max_extent = 0
    all_lines_sensitivity_dict = {}
    line_common_hits = {}
    for client_data in client_data_list:
        max_extent = max(max_extent, client_data['extent'])
        for line, score in client_data['dict'].items():
            if line in all_lines_sensitivity_dict:
                all_lines_sensitivity_dict[line] = max(all_lines_sensitivity_dict[line], score)
            else:
                all_lines_sensitivity_dict[line] = score
            if line in line_common_hits:
                line_common_hits[line] += 1
            else:
                line_common_hits[line] = 1
    sorted_sensitivity_lines = sorted(all_lines_sensitivity_dict.keys(), key=lambda x: all_lines_sensitivity_dict[x], reverse=True)
    
    return client_data_list, max_extent, sorted_sensitivity_lines, line_common_hits

def merge_and_sort_client_data(client_data_list):
    grouped_data = {}
    for data in client_data_list:
        extent = data['extent']
        if extent not in grouped_data:
            grouped_data[extent] = {
                'extent': extent,
                'dict': {}
            }
        for line, score in data['dict'].items():
            if line in grouped_data[extent]['dict']:
                grouped_data[extent]['dict'][line] = max(grouped_data[extent]['dict'][line], score)
            else:
                grouped_data[extent]['dict'][line] = score
    
    merged_list = sorted(grouped_data.values(), key=lambda x: x['extent'])
    return merged_list

def aggregate_palin_ckks_tensor(plain_B,cipherA):
    """
    Aggregate plaintext and CKKS tensors.
    """
    current_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_path, "ckks_full_context.bytes"), "rb") as f:
        context = ts.context_from(f.read())

    assert len(plain_B)==len(cipherA),f"Leng(PlainB){len(plain_B)} is not match Len(CipherA){len(cipherA)}"
    
    result = {}
    layer_num = 0
    for clint_key in plain_B.keys():
        plainBlist = plain_B[clint_key]  
        cipherAlist = pickle.loads(cipherA[clint_key])
        layer_num = len(plainBlist)
        client_res = []
        for index in range(layer_num):
            plain_B_tensor = ts.plain_tensor(plainBlist[index])
            layer_res = []
            for vec_a in cipherAlist[index]:
                enc_col_a = ts.ckks_vector_from(context,vec_a)
                temp = enc_col_a.mm(plain_B_tensor.transpose())  #vector
                layer_res.append(temp)
            client_res.append(layer_res)
        result[clint_key]=client_res
    final_res = []
    layer_nums = len(plain_B.values()[0])
    for layer in range(layer_nums):
        aggs_mix = []
        for mix in result.values():
            aggs_mix.append(mix[layer])
        final_layer = aggregate_ckks_tensors(aggs_mix)
        final_res.append(final_layer)
    return pickle.dumps(final_res)
            
#######################
def process_client(clint_key, plain_B, cipherA, current_path):
    plainBlist = plain_B[clint_key]
    cipherAlist = pickle.loads(cipherA[clint_key])
    layer_num = len(plainBlist)
    client_res = []

    for index in range(layer_num):
        plain_B_tensor = ts.plain_tensor(plainBlist[index])
        layer_res = []

        for vec_a in cipherAlist[index]:
            with open(os.path.join(current_path, "ckks_full_context.bytes"), "rb") as f:
                context = ts.context_from(f.read())
            enc_col_a = ts.ckks_vector_from(context, vec_a)
            temp = enc_col_a.mm(plain_B_tensor.transpose())
            layer_res.append(temp.serialize()) 
        client_res.append(layer_res) 

    return clint_key, client_res 


def parallel_processing_palin_ckks(plain_B, cipherA):
    result = {}
    current_path = os.path.dirname(os.path.abspath(__file__))
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_client, clint_key, plain_B, cipherA, current_path): clint_key
            for clint_key in plain_B.keys()
        }
        for future in futures:
            clint_key, client_res = future.result()
            result[clint_key] = client_res
    final_res = []
    layer_nums = len(next(iter(plain_B.values())) if plain_B else None)
    for layer in range(layer_nums):
        aggs_mix = []
        for mix in result.values():
            aggs_mix.append(mix[layer])
        final_layer = aggregate_ckks_tensors(aggs_mix)
        final_res.append(final_layer)
    return pickle.dumps(final_res)