from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from flwr_datasets import FederatedDataset

FDS = None  # Cache FederatedDataset



def get_tokenizer_and_data_collator_and_propt_formatting(model_name: str,dataset_name:str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    response_template_with_context = "\n### Response:"  # alpaca response tag
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )
    if dataset_name == 'vicgalle/alpaca-gpt4':
        def formatting_prompts_func(example):
            output_texts = []
            mssg = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            for i in range(len(example["instruction"])):
                text = f"{mssg}\n### Instruction:\n{example['instruction'][i]}\n### Response: {example['response'][i]}"
                output_texts.append(text)
            return output_texts
        return tokenizer, data_collator, formatting_prompts_func
    elif dataset_name == 'openai/gsm8k':
        def formatting_prompts_gsm8k(example):
            output_texts = []
            for i in range(len(example["question"])):
                text = f"### Question:\n{example['question'][i]}\n### Response: {example['response'][i]}"
                output_texts.append(text)
            return output_texts
        return tokenizer, data_collator, formatting_prompts_gsm8k
    elif dataset_name == 'Muennighoff/natural-instructions':
        def formatting_prompts_natural_ins(example):
            output_texts = []
            mssg = "In this task, you're given passages that contain mentions of names of people, places, or things. Some of these mentions refer to the same person, place, or thing. Your job is to write questions that evaluate one's understanding of such references. Good questions are expected to link pronouns (she, her, him, his, their, etc.) or other mentions to people, places, or things to which they may refer. Do not ask questions that can be answered correctly without understanding the paragraph or having multiple answers. Avoid questions that do not link phrases referring to the same entity. For each of your questions, the answer should be one or more phrases in the paragraph, and it should be unambiguous."
            for i in range(len(example['inputs'])):
                text = f"{mssg}\n### Instruction:\n{example['inputs'][i]}\n### Response: {example['response'][i]}"
                output_texts.append(text)
            return output_texts
        return tokenizer, data_collator, formatting_prompts_natural_ins



def load_data(partition_id: int, num_partitions: int, dataset_name: str):
    """Load partition data."""
    global FDS
    subname = 'main' if dataset_name=='openai/gsm8k' else None
    if FDS is None:
        if dataset_name == "Muennighoff/natural-instructions":
            FDS = FederatedDataset(
                dataset=dataset_name,
                partitioners={
                "train": DirichletPartitioner(
                    num_partitions=num_partitions,
                    partition_by="task_name",
                    alpha=10,
                    seed=42,
                    min_partition_size=0,),
                    },
                )
        else:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            FDS = FederatedDataset(
                dataset=dataset_name,
                subset= subname,
                partitioners={"train": partitioner},
                )
        
    client_trainset = FDS.load_partition(partition_id, "train")
    if dataset_name == "Muennighoff/natural-instructions":
        import random
        nums_train_sample = 5000
        indices = random.sample(range(len(client_trainset)), nums_train_sample)
        client_trainset = client_trainset.select(indices)

    input_keys = {
        "vicgalle/alpaca-gpt4":"output",
        "openai/gsm8k":"answer",
        "Muennighoff/natural-instructions":"targets",
    }

    client_trainset = client_trainset.rename_column(input_keys[dataset_name], "response")

    return client_trainset

def load_glue_data(partition_id: int, num_partitions: int, dataset_name: str):
    """Load partition data."""
    global FDS
    if FDS is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS = FederatedDataset(
            dataset="glue",
            subset=dataset_name,
            partitioners={"train": partitioner},
        )
    client_trainset = FDS.load_partition(partition_id, "train")

    return client_trainset

def load_imdb_data(partition_id: int, num_partitions: int, dataset_name: str):
    """Load partition data."""
    global FDS
    if FDS is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )
    client_trainset = FDS.load_partition(partition_id, "train")

    return client_trainset


def replace_keys(input_dict, match="-", target="_"):
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
