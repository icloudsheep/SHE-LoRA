#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   gen_ckks_keys.py
@Time    :   2025/03/08 22:05:13
@Author  :   Jianmin Liu 
@Version :   1.0
@Site    :   https://jianmin.cc
@Desc    :   Gen CKKS keys
@Useful_Links:https://github.com/OpenMined/TenSEAL/blob/29eb3d6452202775be7aec37aa0516e1e4e16dba/tutorials/Tutorial%203%20-%20Benchmarks.ipynb#L54
'''

import tenseal as ts
import numpy as np
import os
current_path = os.path.dirname(os.path.abspath(__file__))

# ----------------------------
# Generate CKKS context and save keys
# ----------------------------
def generate_and_save_keys():
    poly_modulus_degree = 8192          # polynomial degree 16384 for 30B                8192 for 7B
    coeff_mod_bit_sizes = [60,40,60]      # coefficient modulus chain [60,40,60]
    global_scale = 2**24                 # encoding scale factor 2**40 for 30B	                2**24 for 7B

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes
    )
    context.generate_galois_keys()
    context.global_scale = global_scale  

    with open(os.path.join(current_path, "ckks_full_context.bytes"), "wb") as f:
        if context.is_private():
            # print("Context contains private key")
            f.write(context.serialize(save_secret_key=True))
        else:
            print("Context does not contain private key")
        
    
    public_context = context.copy()
    public_context.make_context_public()
    with open(os.path.join(current_path, "ckks_public.bytes"), "wb") as f:
        if context.is_private():
            # print("Context contains private key")
            pass
        else:
            # print("Context does not contain private key")
            f.write(public_context.serialize())  
    
    print("CKKS keys have been saved to files: ckks_full_context.bytes (private key) and ckks_public.bytes (public key)")


generate_and_save_keys()