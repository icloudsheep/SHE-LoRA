#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2024/12/27 14:56:21
@Author  :   Jianmin Liu 
@Version :   1.0
@Site    :   https://jianmin.cc
@Desc    :   Client side functions for SHE-LoRA.
'''

import numpy as np
import pickle
import sys
import os
import tenseal as ts



def decrypt(enc,secret_key=None):
    '''
    Decrypt CKKS ciphertext to plaintext.
    '''
    return enc.decrypt(secret_key).tolist()

def exchange_columns(A, selected_cols=None, num_enc_col=None):
    """
    Randomly select `num_enc_col` columns from `A` and swap their positions.
    The swap is reversible; calling this function again will restore the original order.
    For the i-th column swap, it is placed in the position of the (-i+1)-th column:
        - First iteration (i=0): swap column 0 with column -1 (last column)
        - Second iteration (i=1): swap column 1 with column -2 (second-to-last column)
        - Third iteration (i=2): swap column 2 with column -3 (first column)
    """
    if selected_cols ==None and num_enc_col:
        selected_cols = np.random.choice(A.shape[1], num_enc_col, replace=False)
    # Swap columns
    A_1 = A.copy()
    for i, col_idx in enumerate(selected_cols):
        A_1[:, [col_idx, -(i+1)]] = A_1[:, [-(i+1), col_idx]]
    return A_1



def merge_plain_he_2_ab(plain_a,plain_b,selected_cols,rank):
    '''
    plain_a: plain matrix;
    plain_b: decryption from server he. decrypted at local. here only use its results
    '''
    Ua, sa, Vha = np.linalg.svd(plain_b, full_matrices=False)

    B_p , A_p = plain_a[0],plain_b[1] 
    B_e , A_e = Ua @ np.diag(sa), Vha

    new_B = np.hstack((B_p, B_e))
    new_A = np.vstack((A_p, A_e))
    new_A = exchange_columns(new_A, selected_cols)  

    U1 , s1, V1 = np.linalg.svd(new_B, full_matrices=False)
    U2 , s2, V2 = np.linalg.svd(new_A, full_matrices=False)

    result_A = U1 @ np.diag(s1) @ V1 @ U2 @ np.diag(s2)
    result_A = new_B @ U2 @ np.diag(s2)
    result_B = V2
    return result_A, result_B


_CKKS_CONTEXT = None


def _get_ckks_context():
    """Load CKKS context once and cache it."""
    global _CKKS_CONTEXT
    if _CKKS_CONTEXT is None:
        current_path = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_path, "ckks_full_context.bytes"), "rb") as f:
            _CKKS_CONTEXT = ts.context_from(f.read())
    return _CKKS_CONTEXT


def encrypt_cipher_list(cipher_list):
    """
    Encrypt columns using CKKS. Context is loaded once and reused.
    cipher_list: list of numpy arrays, each shape (r, num_cols_to_encrypt).
    """
    context = _get_ckks_context()  # Load once, reuse
    ckks_results = []
    for cipher in cipher_list:
        column_vectors = [cipher[:, i] for i in range(cipher.shape[1])]
        encrypted_layer = [ts.ckks_vector(context, vec).serialize() for vec in column_vectors]
        ckks_results.append(encrypted_layer)
    return pickle.dumps(ckks_results)


def encrypt_column_tensor(vec):
    """Encrypt a single column vector (for backward compat)."""
    context = _get_ckks_context()
    cipher = ts.ckks_vector(context, vec)
    return cipher.serialize()
