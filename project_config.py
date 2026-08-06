from pathlib import Path

from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
SUPPORTED_LLM_VARIANTS = {"openllama-3b-v2", "openllama-7b-v2"}
SUPPORTED_DOLLY_FORMATS = {"auto", "json", "parquet", "csv"}


def require_local_path(value: str, field: str, directory: bool = False) -> str:
    if not value or not str(value).strip():
        raise ValueError(
            f"{field} is empty. Set it to a local path in {CONFIG_PATH}."
        )
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    if directory and not path.is_dir():
        raise NotADirectoryError(f"{field} must be a directory: {path}")
    return str(path)


def load_project_config() -> DictConfig:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Project configuration not found: {CONFIG_PATH}")
    return OmegaConf.load(CONFIG_PATH)


def load_llm_config(validate_paths: bool = True) -> DictConfig:
    root_cfg = load_project_config()
    cfg = OmegaConf.create(OmegaConf.to_container(root_cfg.llm, resolve=True))
    active = cfg.model.active
    if active not in SUPPORTED_LLM_VARIANTS:
        raise ValueError(
            "llm.model.active must be openllama-3b-v2 or openllama-7b-v2."
        )
    if active not in cfg.model.variants:
        raise ValueError(f"Unknown llm.model.active value: {active}")
    if cfg.model.task_type != "NLG":
        raise ValueError("Only the NLG task type is supported by the local LLM setup.")
    if cfg.dataset.name != "dolly":
        raise ValueError("Only the local Dolly dataset is supported.")
    if cfg.dataset.format not in SUPPORTED_DOLLY_FORMATS:
        raise ValueError(
            f"Unsupported llm.dataset.format value: {cfg.dataset.format}"
        )
    if not cfg.model.local_files_only:
        raise ValueError("llm.model.local_files_only must remain true.")

    variant = cfg.model.variants[active]
    if validate_paths:
        variant.path = require_local_path(
            variant.path, f"llm.model.variants.{active}.path", directory=True
        )
        cfg.dataset.path = require_local_path(
            cfg.dataset.path, "llm.dataset.path"
        )

    cfg.model.path = variant.path
    cfg.model.name = variant.path
    cfg.model.display_name = variant.display_name
    cfg.model.architecture = variant.architecture
    return cfg


def load_ckks_benchmark_config(validate_paths: bool = True) -> DictConfig:
    cfg = OmegaConf.create(
        OmegaConf.to_container(
            load_project_config().llm.ckks_benchmark, resolve=True
        )
    )
    ratios = [float(ratio) for ratio in cfg.encryption_ratios]
    if not ratios:
        raise ValueError("llm.ckks_benchmark.encryption_ratios cannot be empty.")
    if any(ratio < 0.0 or ratio > 1.0 for ratio in ratios):
        raise ValueError(
            "llm.ckks_benchmark.encryption_ratios must be between 0 and 1."
        )
    if len(set(ratios)) != len(ratios):
        raise ValueError(
            "llm.ckks_benchmark.encryption_ratios cannot contain duplicates."
        )
    if int(cfg.repeats) < 1:
        raise ValueError("llm.ckks_benchmark.repeats must be at least 1.")
    for field in (
        "max_clients",
        "max_tensors",
        "client_workers",
        "server_workers",
    ):
        value = cfg.get(field)
        if value is not None and int(value) < 1:
            raise ValueError(f"llm.ckks_benchmark.{field} must be null or positive.")
    if int(cfg.ciphertext_batch_columns) < 1:
        raise ValueError(
            "llm.ckks_benchmark.ciphertext_batch_columns must be positive."
        )
    if int(cfg.ciphertext_batch_columns) % 4:
        raise ValueError(
            "llm.ckks_benchmark.ciphertext_batch_columns must be divisible by 4."
        )
    if not str(cfg.input_glob).strip():
        raise ValueError("llm.ckks_benchmark.input_glob cannot be empty.")
    if cfg.run_name:
        run_name = str(cfg.run_name)
        if run_name in {".", ".."} or Path(run_name).name != run_name:
            raise ValueError(
                "llm.ckks_benchmark.run_name must be a single path name."
            )

    cfg.encryption_ratios = ratios
    needs_context = any(ratio > 0.0 for ratio in ratios)
    if validate_paths:
        cfg.input_dir = require_local_path(
            cfg.input_dir, "llm.ckks_benchmark.input_dir", directory=True
        )
        if needs_context:
            cfg.full_context_path = require_local_path(
                cfg.full_context_path, "llm.ckks_benchmark.full_context_path"
            )
        else:
            path = Path(str(cfg.full_context_path)).expanduser()
            cfg.full_context_path = str(
                path if path.is_absolute() else PROJECT_ROOT / path
            )
    else:
        for field in ("input_dir", "full_context_path"):
            path = Path(str(cfg[field])).expanduser()
            cfg[field] = str(path if path.is_absolute() else PROJECT_ROOT / path)

    for field in ("output_dir", "tensorboard_dir"):
        path = Path(str(cfg[field])).expanduser()
        cfg[field] = str((path if path.is_absolute() else PROJECT_ROOT / path).resolve())
    return cfg


def load_vision_config(validate_paths: bool = True) -> DictConfig:
    root_cfg = load_project_config()
    vision_cfg = root_cfg.vision
    active = vision_cfg.active_profile
    if active not in vision_cfg.profiles:
        raise ValueError(f"Unknown vision.active_profile value: {active}")

    cfg = OmegaConf.merge(vision_cfg.common, vision_cfg.profiles[active])
    cfg.active_profile = active
    if not cfg.model.local_files_only:
        raise ValueError("vision.common.model.local_files_only must remain true.")
    if cfg.dataset.download:
        raise ValueError("vision.common.dataset.download must remain false.")
    if validate_paths:
        cfg.model.path = require_local_path(
            cfg.model.path, "vision.common.model.path", directory=True
        )
        cfg.dataset.root = require_local_path(
            cfg.dataset.root, "vision.common.dataset.root", directory=True
        )

    cfg.model.model_name_or_path = cfg.model.path
    cfg.data_root = cfg.dataset.root
    cfg.download_datasets = False
    return cfg
