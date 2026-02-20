"""flowertune-llm: A Flower / FlowerTune app."""

import os
from datetime import datetime

from flwr.common import Context, ndarrays_to_parameters
from flwr.common.config import unflatten_dict
from flwr.server import ServerConfig
from .lib.server_app import ServerApp
from .lib.serverapp_components import ServerAppComponents
from .strategy import FedAvg
from omegaconf import DictConfig

from flowertune_llm.models import get_model, get_parameters, set_parameters
from flowertune_llm.dataset import replace_keys

import pickle
from .utils import (
    set_enc_lines_to_client, #
    plain_adaptive_rank,
    fusion_plain_cipher
)

# Get function that will be executed by the strategy's evaluate() method
# Here we use it to save global model checkpoints
def get_evaluate_fn(cfg, save_every_round, total_round, save_path):
    """Return an evaluation function for saving global model."""

    def evaluate(server_round: int, parameters, config):
        # Save model 
        if server_round not in [0,1,2] and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            # Init model
            model_cfg = cfg.model
            model_cfg.lora.peft_lora_r = config['lora_rank']
            model = get_model(model_cfg)
            global_enc_lines = pickle.loads(config['enc_lines'])
            enc_lines = set_enc_lines_to_client(global_enc_lines,config['he_budget'])
            plain_agg_results = plain_adaptive_rank(parameters,config['lora_rank'])
            new_parameters = plain_agg_results
            set_parameters(model, new_parameters) 
            # model.save_pretrained(f"{save_path}/{cfg.model.name.split("/")[1]}/peft_round-{server_round}_rank-{config['lora_rank']}_HE-{config['he_budget']}_leak")
            cipher_agg_results = config['ckks']

            new_parameters = fusion_plain_cipher(plain_agg_results,cipher_agg_results,enc_lines,config["lora_rank"])
            set_parameters(model, new_parameters) 
            # model.save_pretrained(f"{save_path}/{cfg.model.name.split("/")[1]}/peft_round-{server_round}_rank-{config['lora_rank']}_HE-{config['he_budget']}")

        return 0.0, {}

    return evaluate


def get_on_fit_config(save_path):
    """Return a function that will be used to construct the config that the client's
    fit() method will receive."""

    def fit_config_fn(server_round: int):
        fit_config = {}
        fit_config["current_round"] = server_round
        fit_config["save_path"] = save_path
        fit_config["he_budget"] = 16 
        fit_config["lora_rank"] = 32
        fit_config['enc_lines'] = b'' 
        fit_config['ckks'] = b''

        return fit_config

    return fit_config_fn


def fit_weighted_average(metrics):
    """Aggregate (federated) evaluation metrics."""
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"train_loss": sum(losses) / sum(examples)}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"results/{folder_name}")
    os.makedirs(save_path, exist_ok=True)

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    # INIT WANDB config
    # wandb.init(entity="laujianmin-ylab",project='flwr-simu-local', name=time.strftime('%m%d%H%M%S'),config=cfg)
    proj_name = "SHE-LoRA "+ "-"+ cfg.model.name.split("/")[1] + "-"+cfg.dataset.name.split("/")[1] 
    # proj_name = "Heter"+ "-"+ "SHE-LoRA "+ "-"+ cfg.model.name.split("/")[1] 
    
    # INIT WANDB config
    wandb.init(entity="laujianmin-ylab",project=proj_name, name=time.strftime('%m%d%H%M%S'),config=cfg)

    # print(cfg.model.lora.peft_lora_r)
    # print(type(cfg.model.lora.peft_lora_r)) # ListConfig
    lora_ranks = list(cfg.model.lora.peft_lora_r)
    lora_he_budgets = list(cfg.model.lora.he_budget)
    config_pairs = list(zip(lora_ranks, lora_he_budgets))  
       
    max_rank_of_system = max(list(cfg.model.lora.peft_lora_r))
    cfg.model.lora.peft_lora_r = max_rank_of_system
    # print("Server Init Model of Rank: ",cfg.model.lora)
    # Get initial model weights
    init_model = get_model(cfg.model)
    init_model_parameters = get_parameters(init_model)
    init_model_parameters = ndarrays_to_parameters(init_model_parameters)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=cfg.strategy.fraction_fit,   
        fraction_evaluate=cfg.strategy.fraction_evaluate,   
        on_fit_config_fn=get_on_fit_config(save_path),    
        fit_metrics_aggregation_fn=fit_weighted_average,   
        initial_parameters=init_model_parameters,         
        evaluate_fn=get_evaluate_fn(                    
            cfg, cfg.train.save_every_round, num_rounds, save_path
        ),
        config_pairs = config_pairs,
        max_rank_of_system=max_rank_of_system
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

import wandb
import time
import os
# os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_MODE"] = "disabled"
# Flower ServerApp
app = ServerApp(server_fn=server_fn)
wandb.finish()