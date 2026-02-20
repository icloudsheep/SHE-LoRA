# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Local DP modifier."""


from logging import INFO

import numpy as np

from flwr.client.typing import ClientAppCallable
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common import recordset_compat as compat
from flwr.common.constant import MessageType
from flwr.common.context import Context
from flwr.common.differential_privacy import (
    add_localdp_gaussian_noise_to_params,
    compute_clip_model_update,
)
from flwr.common.logger import log
from flwr.common.message import Message
from ..utils import plain_adaptive_rank

class LocalDpMod:
    """Modifier for local differential privacy.

    This mod clips the client model updates and
    adds noise to the params before sending them to the server.

    It operates on messages of type `MessageType.TRAIN`.

    Parameters
    ----------
    clipping_norm : float
        The value of the clipping norm.
    sensitivity : float
        The sensitivity of the client model.
    epsilon : float
        The privacy budget.
        Smaller value of epsilon indicates a higher level of privacy protection.
    delta : float
        The failure probability.
        The probability that the privacy mechanism
        fails to provide the desired level of privacy.
        A smaller value of delta indicates a stricter privacy guarantee.

    Examples
    --------
    Create an instance of the local DP mod and add it to the client-side mods:

    >>> local_dp_mod = LocalDpMod( ... )
    >>> app = fl.client.ClientApp(
    >>>     client_fn=client_fn, mods=[local_dp_mod]
    >>> )
    """

    def __init__(
        self, clipping_norm: float, sensitivity: float, epsilon: float, delta: float
    ) -> None:
        if clipping_norm <= 0:
            raise ValueError("The clipping norm should be a positive value.")

        if sensitivity < 0:
            raise ValueError("The sensitivity should be a non-negative value.")

        if epsilon < 0:
            raise ValueError("Epsilon should be a non-negative value.")

        if delta < 0:
            raise ValueError("Delta should be a non-negative value.")

        self.clipping_norm = clipping_norm
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta
        self.dp_for = 'A'

    def __call__(
        self, msg: Message, ctxt: Context, call_next: ClientAppCallable
    ) -> Message:
        """Perform local DP on the client model parameters.

        Parameters
        ----------
        msg : Message
            The message received from the server.
        ctxt : Context
            The context of the client.
        call_next : ClientAppCallable
            The callable to call the next middleware in the chain.

        Returns
        -------
        Message
            The modified message to be sent back to the server.
        """

        if msg.metadata.message_type != MessageType.TRAIN:
            return call_next(msg, ctxt)

        fit_ins = compat.recordset_to_fitins(msg.content, keep_input=True)        
        server_to_client_params = parameters_to_ndarrays(fit_ins.parameters)

        # Call inner app
        out_msg = call_next(msg, ctxt)

        import time
        start_time = time.time()
        try:
            message_type = out_msg.content.parameters_records['fitres.parameters']['_empty'].stype
            if message_type == "neo":
                return out_msg
        except KeyError as e:
            pass
        
        # Check if the msg has error
        if out_msg.has_error():
            return out_msg
        
        fit_res = compat.recordset_to_fitres(out_msg.content, keep_input=True)

        client_to_server_params = parameters_to_ndarrays(fit_res.parameters)

        if fit_ins.config["current_round"] != 2:
            client_rank = fit_ins.config["lora_rank"]
            server_to_client_params = plain_adaptive_rank(server_to_client_params,client_rank)
            for index, (client_param, server_param) in enumerate(zip(client_to_server_params, server_to_client_params)):
                if client_param.shape != server_param.shape:
                    print(f"Shape mismatch at index {index}: client shape {client_param.shape}, server shape {server_param.shape}")
            # Clip the client update
            compute_clip_model_update(
                client_to_server_params,
                server_to_client_params,
                self.clipping_norm,
            )
            log(
                INFO,
                "LocalDpMod: parameters are clipped by value: %.4f.",
                self.clipping_norm,
            )

        
        if self.dp_for == 'A':
            matrices_b = [martix for index,martix in enumerate(client_to_server_params) if index % 2==1] 
        elif self.dp_for == 'B':
            matrices_a = [martix for index,martix in enumerate(client_to_server_params) if index % 2==0] 
        else:
            pass # add dp for all 

        fit_res.parameters = ndarrays_to_parameters(client_to_server_params)


        # Add noise to model params
        fit_res.parameters = add_localdp_gaussian_noise_to_params(
            fit_res.parameters, self.sensitivity, self.epsilon, self.delta
        )

        results = parameters_to_ndarrays(fit_res.parameters)
        if self.dp_for == 'A':
            for index, matrix in enumerate(matrices_b):
                results[index * 2 + 1] = matrix
        elif self.dp_for == 'B':
            for index, matrix in enumerate(matrices_a):
                results[index * 2] = matrix
        else:
            pass
        fit_res.parameters = ndarrays_to_parameters(results)

        noise_value_sd = (
            self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        )
        log(
            INFO,
            "LocalDpMod: local DP noise with %.4f stedv added to parameters",
            noise_value_sd,
        )
        end_time = time.time()
        print(f"LocalDpMod: time cost: {end_time - start_time} seconds")
        out_msg.content = compat.fitres_to_recordset(fit_res, keep_input=True)
        return out_msg