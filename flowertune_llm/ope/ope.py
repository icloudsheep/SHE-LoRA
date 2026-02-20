#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   ope.py
@Time    :   2025/01/07 17:40:40
@Author  :   Jianmin Liu 
@Version :   1.0
@Site    :   https://jianmin.cc
@Desc    :   OPE Func
'''


import random
import numpy as np

class OPE:
    def __init__(self, domain_min=0, domain_max=9999, range_min=10000, range_max=99999,seed=0):
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.range_min = range_min
        self.range_max = range_max
        
        domain_size = domain_max - domain_min + 1
        range_size = range_max - range_min + 1
        if range_size < domain_size:
            raise ValueError("Range size must be at least domain size")
        
        if seed is not None:
            random.seed(seed)
    
    def encrypt(self, plaintext):
        if plaintext < self.domain_min or plaintext > self.domain_max:
            raise ValueError("Plaintext out of domain range")
        
        domain_size = self.domain_max - self.domain_min + 1
        range_size = self.range_max - self.range_min + 1
        base, remainder = divmod(range_size, domain_size)
        pos = plaintext - self.domain_min
        
        if pos < remainder:
            interval_start = pos * (base + 1)
            interval_length = base + 1
        else:
            interval_start = remainder * (base + 1) + (pos - remainder) * base
            interval_length = base
        
        cipher_start = self.range_min + interval_start
        cipher_end = cipher_start + interval_length - 1
        
        cipher = random.randint(cipher_start, cipher_end)
        return cipher
    
    def decrypt(self, ciphertext):
        if ciphertext < self.range_min or ciphertext > self.range_max:
            raise ValueError("Ciphertext out of range")
        
        range_size = self.range_max - self.range_min + 1
        domain_size = self.domain_max - self.domain_min + 1
        proportion = (ciphertext - self.range_min) / range_size
        plaintext = int(domain_size * proportion) + self.domain_min
        return plaintext

def get_top_k_elements(arr, k):
    indices = np.argsort(arr)[-k:]
    indices = indices[::-1]
    top_k_values = arr[indices]
    
    ope = OPE()
    result = [{"line": ope.encrypt(int(pos)), "score": float(val)} for pos, val in zip(indices, top_k_values)]
    
    return result


def ope_process_dict_top_k(input_dict, k):
    result_dict = {}
    for key, value in input_dict.items():
        top_k_elements = get_top_k_elements(value, k)
        result_dict[key] = top_k_elements
    return result_dict