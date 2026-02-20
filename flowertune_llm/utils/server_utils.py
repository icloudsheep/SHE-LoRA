#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   server_utils.py
@Time    :   2025/03/04 20:53:58
@Author  :   Jianmin Liu 
@Version :   1.0
@Site    :   https://jianmin.cc
@Desc    :   Tools for server
'''

from flwr.common import FitRes, NDArray, NDArrays, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
import numpy as  np

def get_plainB_cipherA_from_results(results: list[tuple[ClientProxy, FitRes]]):
    plainB,cipher= {},{}

    num_examples_total = sum(fit_res.num_examples for (_, fit_res) in results)
    scaling_factors = np.asarray(
        [fit_res.num_examples / num_examples_total for _, fit_res in results]
    )
    for index,(client,fit_res) in enumerate(results):
        scale = scaling_factors[index]
        params= parameters_to_ndarrays(fit_res.parameters)
        params_B = [par * scale for par in params[1::2]] 
        client_id = client.cid
        plainB[client_id] = params_B[:-1] if params[-1].ndim==1 else params_B 
        ckks = fit_res.metrics['ckks']
        cipher[client_id] = ckks

    return plainB,cipher

