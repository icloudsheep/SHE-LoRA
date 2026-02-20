#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   fedavg.py
@Time    :   2024/12/05 15:58:25
@Author  :   Jianmin Liu 
@Version :   1.0
@Site    :   https://jianmin.cc
@Desc    :   Modify the FedAVG to FLoRA_AVG
'''

# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: arxiv.org/abs/1602.05629
"""

from logging import INFO
from logging import WARNING
from typing import Callable, Optional, Union
import pickle
import time

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    # FitResNeo, 
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from .aggregate import (
    aggregate, 
    aggregate_inplace, 
    weighted_loss_avg,
    aggregate_flora,
)
from ..she.ckks_server import aggregate_palin_ckks_tensor,parallel_processing_palin_ckks
# from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg
from flwr.server.strategy import Strategy
import json
import random
from .negotiation import embed_client_budget, get_nego_enclines_bayesian
from ..utils.server_utils import get_plainB_cipherA_from_results


NEGOTIATION = "Neo"  # Macro scalar, mark the return message of negotiation.
WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

random.seed(0)

# pylint: disable=line-too-long
class FedAvg(Strategy):
    """Federated Averaging strategy.

    Implementation based on https://arxiv.org/abs/1602.05629

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    inplace : bool (default: True)
        Enable (True) or disable (False) in-place aggregation of model updates.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
        config_pairs = [],
        max_rank_of_system = None
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace
        self.stage = 'neo'  # or 'fl'
        self.enc_lines = None
        self.config_pairs = config_pairs 
        self.max_rank_of_system = max_rank_of_system
        self.clients = [] # Client list, add rank and he_budget attributes, and return plaintext and ciphertext based on he_budget when returning.
        self.ckks_bytes = [] # Ciphertext aggregation result, [bytes,bytes,...].

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """ADD CONFIG FOR CKKS"""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        if server_round==1:
            pass
        elif server_round ==2:
            config['enc_lines'] = pickle.dumps(self.enc_lines)
        else:        
            config['enc_lines'] = pickle.dumps(self.enc_lines)
            config['ckks'] = self.ckks_bytes
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        # eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        # random sample clients
        random_index = random.randint(0, len(self.config_pairs)-1)
        client_config = self.config_pairs[random_index]
        rank, budget = client_config
        config["he_budget"] = budget 
        config["lora_rank"] = rank
        
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, config)
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        if server_round==1:
            fit_ins = FitIns(parameters, config)
        elif server_round ==2:
            config['enc_lines'] = pickle.dumps(self.enc_lines)
            fit_ins = FitIns(parameters, config)
        else:        
            config['enc_lines'] = pickle.dumps(self.enc_lines)
            config['ckks'] = self.ckks_bytes
            fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Assign different configurations based on client budget
        client_instructions = []
        
        # Using random.choice can repeat sampling, i.e. the same configuration pair can be assigned to multiple clients
        for client in clients:
            # Check if client already has configuration
            if 'lora_rank' in client.properties and 'he_budget' in client.properties:
                # If client already has configuration, use existing configuration
                rank = client.properties['lora_rank']
                budget = client.properties['he_budget']
                client_config = (rank, budget)
            else:
                # If client does not have configuration, randomly select one from config_pairs
                random_index = random.randint(0, len(self.config_pairs)-1)
                client_config = self.config_pairs[random_index]
                # If config_pairs length is greater than 1, remove selected configuration
                if len(self.config_pairs) > 1:
                    self.config_pairs.pop(random_index)
                rank, budget = client_config  
                client = embed_client_budget(client,rank,budget)
                self.clients.append(client)
            client_specific_fit_ins = FitIns(fit_ins.parameters, fit_ins.config.copy())  # Deep copy configuration
            client_specific_fit_ins.config["lora_rank"] = client_config[0]  # Add rank to configuration
            client_specific_fit_ins.config["he_budget"] = client_config[1]  # Add budget to configuration
            client_instructions.append((client, client_specific_fit_ins))

        return client_instructions

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: Union[list[tuple[ClientProxy, FitRes]], list[tuple[int,dict]]], 
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if self.stage =='neo' :
            self.stage = 'fl' # Next round will no longer negotiate.
            log(
            INFO,
            "Start negotiating common encryption range.",
              )
            self.server_do_negotiation(results)
            return NEGOTIATION,self.enc_lines 
                

        """Aggregate fit results using weighted average."""
        if not results:
            print("if not results：in fed_avg agg_fit")
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            print("if failures：in fed_avg agg_fit")
            return None, {}

        if self.inplace:
            # Does in-place weighted average of results
            log(INFO,"Server Flora aggregate plaintext.")
            start_plain_time = time.time()
            aggregated_ndarrays = aggregate_flora(results,self.max_rank_of_system)  
            end_plain_time = time.time()
            log(INFO,"Plaintext parameter aggregation completed. (%s seconds)",end_plain_time-start_plain_time)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

            log(INFO,"Get plaintext B and ciphertext A.")
            plain_B,cipherA = get_plainB_cipherA_from_results(results) # dict[cid,list]
            log(INFO,"CKKS ciphertext aggregation started.")
            start_ckks_time = time.time()
            self.ckks_bytes = parallel_processing_palin_ckks(plain_B,cipherA) 
            end_ckks_time = time.time()
            log(INFO,"CKKS ciphertext aggregation completed. (%s seconds)",end_ckks_time-start_ckks_time)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated

    def server_do_negotiation(self,results):
        output = {}
        for client,fit_res in results:
            scores = fit_res.metrics['scores']
            scores = scores.decode('utf-8')
            scores = json.loads(scores)

            # Use cid so keys match negotiation (client_budgets / cids_with_budget use cid)
            output[int(client.cid)] = scores
        sorted(self.clients, key=lambda x: x.properties['he_budget'], reverse=True)
        start_time = time.time()
        results = get_nego_enclines_bayesian(self.clients, output)
        end_time = time.time()
        print("Negotiation time: ",end_time-start_time)
        for key,value in results.items():
            if len(value) != len(set(value)):
                print(f"Warning: Duplicate row numbers found in layer {key}")
        self.enc_lines = results
