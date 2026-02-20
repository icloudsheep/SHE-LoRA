"""flowertune-llm: A Flower / FlowerTune app."""

import os
import warnings
from typing import Dict, Tuple

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.config import unflatten_dict
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig

from transformers import (TrainingArguments,
                          DataCollatorWithPadding,
                          Trainer,
                          AutoTokenizer,
                          )
from trl import SFTTrainer

from flowertune_llm.dataset import (
    get_tokenizer_and_data_collator_and_propt_formatting,
    load_data,
    load_glue_data,
    load_imdb_data,
    replace_keys,
)
from flowertune_llm.models import (
    cosine_annealing,
    get_model,
    set_parameters,
    get_parameters,
)
from .mod.local_dp_mod import LocalDpMod
from .wasens.sensitivity import evaluate_lora_importance
from .ope import ope_process_dict_top_k
from .utils import (
    set_enc_lines_to_client, 
    handle_parameters_to_server, 
    plain_adaptive_rank,
    fusion_plain_cipher
)
import json
import pickle
import time
import timeit

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning)

# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
class FlowerClient(NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        trainset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        num_rounds,
        actual_task = None
    ):  # pylint: disable=too-many-arguments
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.training_argumnets = TrainingArguments(**train_cfg.training_arguments)
        self.tokenizer = tokenizer
        self.formatting_prompts_func = formatting_prompts_func
        self.data_collator = data_collator
        self.num_rounds = num_rounds
        self.trainset = trainset
        self.task_type = model_cfg.task_type
        self.actual_task = actual_task

        # instantiate model
        # print("Client init Here")
        self.model_cfg = model_cfg
        self.model = None
        # self.model = get_model(model_cfg)
        # self.model.seqlen = self.model.config.max_position_embeddings 
        # self.negotiation = True  
        self.enc_lines = {} 
        self.he_budget = 0

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        start_client_time = timeit.default_timer()
        log(INFO,"-------New CLIENT_fit------")
        self.model_cfg.lora.peft_lora_r = config['lora_rank']
        self.model = get_model(self.model_cfg)
        self.model.seqlen = self.train_cfg.seq_length 
        self.model.print_trainable_parameters()
        self.he_budget = config["he_budget"]

        
        if config['current_round']==1:
            log(INFO,"--------Negotiation phase------------")
            line_scores, element_scores = evaluate_lora_importance(self.model,self.trainset[:200],self.tokenizer,self.task_type,self.actual_task,self.device)
            # with open(f"./parameter_evals/Element_Importance_qwen_pile_{config['current_round']}.pkl", 'wb') as f:
            #     pickle.dump(element_scores, f)
            #     print("Scores Saved!")
            results = ope_process_dict_top_k(line_scores,self.he_budget)
            json_results = json.dumps(results)
            json_bytes = json_results.encode('utf-8')


            return (
                    None,  
                    self.he_budget,
                    {"scores": json_bytes}, 
                )
        elif config['current_round']==2:          
            global_enc_lines = pickle.loads(config['enc_lines'])
            self.enc_lines = set_enc_lines_to_client(global_enc_lines,self.he_budget)
            parameters = get_parameters(self.model)
        else: 
            global_enc_lines = pickle.loads(config['enc_lines'])
            self.enc_lines = set_enc_lines_to_client(global_enc_lines,self.he_budget)
            if "bert" in self.model_cfg.name:
                weight_bias = parameters[-2:]
                parameters = parameters[:-2]
            plain_agg_results = plain_adaptive_rank(parameters,config['lora_rank'])
            cipher_agg_results = config['ckks']

            new_parameters = fusion_plain_cipher(plain_agg_results,cipher_agg_results,self.enc_lines,config["lora_rank"])
            if "bert" in self.model_cfg.name:
                new_parameters.extend(weight_bias)
            set_parameters(self.model, new_parameters) 

        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        self.training_argumnets.learning_rate = new_lr
        self.training_argumnets.output_dir = config["save_path"]

        # # Construct trainer
        trainer = self.trainer_loader()

        # Do local training
        results = trainer.train()

        # if config['lora_rank']==64:
        #     self.model.save_pretrained(f"./results/SHE-LoRA/{config['lora_rank']}/peft_round-{config['current_round']}_rank-{config['lora_rank']}")

        start_time = time.time()
        plain_parameters, cipher_parameters = handle_parameters_to_server(self.model, self.enc_lines, self.he_budget)
        end_time = time.time()
        he_enc_time = end_time - start_time
        log(
            INFO,
            f"Client encryption time: {he_enc_time} seconds"
        )

        end_client_time = timeit.default_timer()
        client_run_all_time = end_client_time-start_client_time
        log(INFO,
            f"Clent Local Train Costs {client_run_all_time}s")
        # return (plain_parameters,len(self.trainset), {"train_loss":0,"he_enc_time":he_enc_time,"run_time":client_run_all_time,"ckks": cipher_parameters})
        return (plain_parameters,len(self.trainset), {"train_loss":results.training_loss,"he_enc_time":he_enc_time,"run_time":client_run_all_time,"ckks": cipher_parameters})
    
    def trainer_loader(self):
        if self.task_type == 'NLG':
            trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_argumnets,
            max_seq_length=self.train_cfg.seq_length,
            train_dataset=self.trainset,
            formatting_func=self.formatting_prompts_func,
            data_collator=self.data_collator,
            )
        elif self.task_type == 'NLU':
            task_to_keys = {
            "cola": ("sentence", None),
            "mnli": ("premise", "hypothesis"),
            "mnli": ("premise", "hypothesis"),
            "mrpc": ("sentence1", "sentence2"),
            "qnli": ("question", "sentence"),
            "qqp": ("question1", "question2"),
            "rte": ("sentence1", "sentence2"),
            "sst2": ("sentence", None),
            "stsb": ("sentence1", "sentence2"),
            "wnli": ("sentence1", "sentence2"),
            "stanfordnlp/imdb":("text", None)
        }
            def preprocess_function(examples):
                sentence1_key, sentence2_key = task_to_keys[self.actual_task]
                if sentence2_key is None:
                    return self.tokenizer(examples[sentence1_key], truncation=True, padding="max_length",return_tensors="pt")
                return self.tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding="max_length",return_tensors="pt")
            tokenized_datasets = self.trainset.map(preprocess_function, batched=True)
            if self.actual_task =="stanfordnlp/imdb":
                remove_columns = list(task_to_keys[self.actual_task])
            else:
                remove_columns = ['idx'] + list(task_to_keys[self.actual_task])
            if None in remove_columns:
                remove_columns.remove(None)
            tokenized_datasets = tokenized_datasets.remove_columns(remove_columns)
            tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
            tokenized_datasets.set_format('torch')

            trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_argumnets,
            train_dataset=tokenized_datasets.select(range(100)),
            data_collator=self.data_collator,
            )
        else:
            raise NotImplementedError("Unsupported task")
        
        return trainer
        


def client_fn(context: Context) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    if cfg.model.task_type == "NLG":
        client_trainset = load_data(partition_id, num_partitions, cfg.dataset.name)
        (
            tokenizer,
            data_collator,
            formatting_prompts_func,
        ) = get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name,cfg.dataset.name)

        return FlowerClient(
            cfg.model,
            cfg.train,
            client_trainset,
            tokenizer,
            formatting_prompts_func,
            data_collator,
            num_rounds,
            actual_task=cfg.dataset.name
        ).to_client()
    elif cfg.model.task_type == "NLU":
        task = cfg.dataset.name
        actual_task = task
        if actual_task in []:
            client_trainset = load_glue_data(partition_id, num_partitions, cfg.dataset.name)
        elif actual_task == "stanfordnlp/imdb":
            client_trainset = load_imdb_data(partition_id,num_partitions,actual_task)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    

        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)

        return FlowerClient(
            cfg.model,
            cfg.train,
            client_trainset,
            tokenizer,
            None,
            data_collator,
            num_rounds,
            actual_task
        ).to_client()


from flwr.common.logger import log
from logging import INFO
# Flower ClientApp
# local_dp_mod = LocalDpMod(clipping_norm=3,sensitivity=1,epsilon=10,delta=1e-5)
# app = ClientApp(client_fn,mods=[local_dp_mod])
app = ClientApp(client_fn)
