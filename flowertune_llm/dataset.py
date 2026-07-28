from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM


_DATASET_CACHE = {}


def get_tokenizer_and_data_collator_and_propt_formatting(model_path: str, dataset_cfg):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        padding_side="right",
        local_files_only=True,
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
    instruction_column = dataset_cfg.instruction_column
    context_column = dataset_cfg.context_column
    response_column = dataset_cfg.response_column

    def formatting_prompts_func(example):
        output_texts = []
        contexts = example.get(context_column, [""] * len(example[instruction_column]))
        for instruction, context, response in zip(
            example[instruction_column], contexts, example[response_column]
        ):
            context_text = f"\n### Context:\n{context}" if context else ""
            output_texts.append(
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request."
                f"\n### Instruction:\n{instruction}{context_text}"
                f"\n### Response: {response}"
            )
        return output_texts

    return tokenizer, data_collator, formatting_prompts_func


def _load_local_dataset(dataset_cfg) -> Dataset:
    path = Path(dataset_cfg.path)
    cache_key = (str(path), dataset_cfg.split, dataset_cfg.format)
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    if path.is_dir():
        dataset = load_from_disk(str(path))
        if isinstance(dataset, DatasetDict):
            if dataset_cfg.split not in dataset:
                raise KeyError(
                    f"Split '{dataset_cfg.split}' is not present in {path}."
                )
            dataset = dataset[dataset_cfg.split]
    else:
        dataset_format = dataset_cfg.format
        if dataset_format == "auto":
            extensions = {
                ".json": "json",
                ".jsonl": "json",
                ".parquet": "parquet",
                ".csv": "csv",
            }
            dataset_format = extensions.get(path.suffix.lower())
        if dataset_format not in {"json", "parquet", "csv"}:
            raise ValueError(f"Unsupported local dataset format for {path}.")
        dataset = load_dataset(
            dataset_format,
            data_files={dataset_cfg.split: str(path)},
            split=dataset_cfg.split,
        )

    required_columns = {
        dataset_cfg.instruction_column,
        dataset_cfg.response_column,
    }
    missing = required_columns.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Dolly dataset is missing columns: {sorted(missing)}")
    _DATASET_CACHE[cache_key] = dataset
    return dataset


def load_data(partition_id: int, num_partitions: int, dataset_cfg):
    dataset = _load_local_dataset(dataset_cfg)
    if not 0 <= partition_id < num_partitions:
        raise ValueError(
            f"partition_id must be in [0, {num_partitions}), got {partition_id}."
        )
    return dataset.shard(
        num_shards=num_partitions, index=partition_id, contiguous=True
    )
