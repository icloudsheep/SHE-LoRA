# SHE-LoRA

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue)](https://iclr.cc/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

<p align="center">
  <img src="_static/banner.jpg" alt="SHE-LoRA Banner" width="100%">
</p>

<!-- start intro -->

This repository provides the implementation of the paper [**SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA**](https://openreview.net/forum?id=PWChrnrw7Z) (ICLR 2026). SHE-LoRA integrates **selective homomorphic encryption (SHE)** and **low-rank adaptation (LoRA)** to enable efficient, privacy-preserving federated fine-tuning of large language models (LLMs) in cross-device settings. Based on parameter sensitivity assessment, heterogeneous clients negotiate a global HE subset and selectively encrypt the most sensitive LoRA parameters; column-swapping and column-aware aggregation keep ciphertext size small while supporting heterogeneous client ranks and encryption budgets. Compared with full-HE baselines, SHE-LoRA reduces **communication overhead by ~99.71%** and **encryption time by ~99.87%**, maintains model performance comparable to non-private baselines, and achieves strong resistance to gradient inversion attacks (e.g., [DAGER](https://github.com/insait-institute/dager-gradient-inversion)) and [membership inference attacks](https://github.com/Nikkei/fast-mia).

<table>
  <tr>
    <td width="50%"><img src="_static/encryption_time.jpg" width="396" alt="Encryption time"></td>
    <td width="50%"><img src="_static/communication_overhead.jpg" width="396" alt="Communication overhead"></td>
  </tr>
  <tr>
    <td width="50%">Encryption time per round.</td>
    <td width="50%">Communication overhead per round.</td>
  </tr>
  <tr>
    <td width="50%"><img src="_static/dager_resistance.jpg" width="396" alt="DAGER resistance"></td>
    <td width="50%"><img src="_static/mia_resistance.jpg" width="396" alt="MIA resistance"></td>
  </tr>
  <tr>
    <td width="50%">Data reconstruction scores under DAGER attack (lower is better).</td>
    <td width="50%">Membership inference attacks (MIA) results.</td>
  </tr>
</table>

SHE-LoRA consists of the following four key components:

<p align="center">
<img src="_static/SHE-LoRA.jpg" align="center" width="90%" alt="SHE-LoRA workflow (placeholder)"/>
</p>

<!-- end intro -->

## 1. HE Subset Negotiation

<!-- start negotiation -->

Clients assess parameter importance (Wanda-style sensitivity) and encode HE subset positions with order-preserving encryption (OPE); the server negotiates a global HE subset and returns it (e.g., as `enc_lines`) so that ciphertext size does not grow with the union of all client masks. Relevant code:

- [flowertune_llm/strategy/negotiation.py](flowertune_llm/strategy/negotiation.py): server-side negotiation (Common list, Sensitivity list, global HE subset).
- [flowertune_llm/ope/ope.py](flowertune_llm/ope/ope.py): OPE for encoding client HE subsets.
- [flowertune_llm/wasens/sensitivity.py](flowertune_llm/wasens/sensitivity.py): parameter sensitivity (Wanda-style) for selecting important columns.

<!-- end negotiation -->

## 2. Selective Encryption of Model Parameter Matrix

<!-- start encryption -->

Each client swaps columns so that columns to encrypt are clustered and aligned (e.g., last k_i columns), then encrypts only those columns with CKKS in blocks and sends the plain adapter part and cipher blocks to the server. Relevant code:

- [flowertune_llm/she/ckks_client.py](flowertune_llm/she/ckks_client.py): column exchange, encrypt, decrypt.
- [flowertune_llm/utils/client_utils.py](flowertune_llm/utils/client_utils.py): `handle_parameters_to_server`, `set_enc_lines_to_client`.

<!-- end encryption -->

## 3. Adaptive Aggregation

<!-- start aggregation -->

The server aggregates plain parts with column-wise weighted averaging and cipher parts in the encrypted domain; it sends SVD-sliced plain results and aggregated ciphertext to clients. Relevant code:

- [flowertune_llm/strategy/fedavg.py](flowertune_llm/strategy/fedavg.py): FedAvg with plain/cipher aggregation and config per client.
- [flowertune_llm/strategy/aggregate.py](flowertune_llm/strategy/aggregate.py): `aggregate_flora`, plain/cipher splitting.
- [flowertune_llm/she/ckks_server.py](flowertune_llm/she/ckks_server.py): server-side CKKS aggregation.

<!-- end aggregation -->

## 4. Reparameterization

<!-- start reparameterization -->

Each client decrypts its cipher part, applies SVD and zero-padding, merges plain and cipher LoRA updates by Eq.(4) in the paper, then reparameterizes to its local rank for the next round. Relevant code:

- [flowertune_llm/utils/client_utils.py](flowertune_llm/utils/client_utils.py): `fusion_plain_cipher`, `plain_adaptive_rank`.

<!-- end reparameterization -->

<!-- start run -->

## How to Use

### 1. Deployment
The main entry for local deployment is [pythonic_starter.py](pythonic_starter.py), or you can use the Flower CLI. The code can be executed with [start.sh](start.sh) or as follows.

**Clone the repository (with submodules, if any):**

```bash
git clone --recurse-submodules https://github.com/liyan2015/SHE-LoRA.git
cd SHE-LoRA
```

If the repository was already cloned without submodules, initialize and update them:

```bash
git submodule update --init --recursive
```

**Create a Conda environment from exported config (recommended for reproducibility):**

The `environment.yml` in this repo is exported from the `she-lora` Conda environment. To reproduce the environment:

```bash
conda env create -f environment.yml
conda activate she-lora
```

Then install the `flwr` lib in editable mode:

```bash
cd ./flower
pip install -e .
```

### 2. Run SHE-LoRA
**Generate HE keys**

```bash
python ./flowertune_llm/she/gen_ckks_keys.py
```

**Run with Flower CLI (recommended):**

```bash
flwr run .
```
---

**Run with local script (optional):**

```bash
python pythonic_starter.py
```

**Override config from command line (optional):**

```bash
# Different model and quantization
flwr run . --run-config "model.name='openlm-research/open_llama_7b_v2' model.quantization=8"

# Fewer rounds, different participation rate
flwr run . --run-config "num-server-rounds=50 strategy.fraction-fit=0.25"
```

Main configurations (model, dataset, LoRA rank, HE budget, rounds, etc.) are in [pyproject.toml](pyproject.toml) under `[tool.flwr.app.config]` and `[tool.flwr.federations.local-simulation]`. Ensure `model.lora.peft-lora-r` and `model.lora.he-budget` list lengths match the number of client types.

**Run the CLIP (vision) version:**

```bash
# Preprocess datasets (required only for DTD and EuroSAT)
python -m vision.scripts.setup_dtd_dataset
python -m vision.scripts.setup_eurosat_dataset

# Run federated CLIP with SHE; choose one config per run
python -m vision.federated_clip_she federated_mnist.yaml
```
<!-- end run -->

## Prerequisites and environment configuration

To run the code, the following are needed:

- **Python** ≥ 3.10 (we use 3.12 in the exported `she-lora` env)
- **CUDA** 11.8 or 12.x (for PyTorch and bitsandbytes)
- **TenSEAL** for CKKS (included in `environment.yml` via pip)
- **Flower**, **PEFT**, **transformers**, **trl**, **bitsandbytes**, **omegaconf**, etc. (see `pyproject.toml` and `environment.yml`)



## Citing

<!-- start citation -->

If you find this repository useful, please consider citing it:

```bibtex
@INPROCEEDINGS{SHE-LoRA,
	author={Liu, Jianmin and Yan, Li and Li, Borui and Yu, Lei and Shen, Chao},
	title={{SHE}-Lo{RA}: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous Lo{RA}},
	booktitle={Proc. of ICLR},
	year={2026},
}
```
List of publications that cite this work: [Google Scholar](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=13481755242758523388)

<!-- end citation -->

## License and acknowledgments

This project is licensed under **Apache-2.0**.  We build on [<img src="https://flower.dev/favicon.ico" width="16" height="16" alt="Flower" /> Flower](https://flower.dev/), [<img src="https://huggingface.co/favicon.ico" width="16" height="16" alt="PEFT" /> PEFT](https://huggingface.co/docs/peft), [<img src="https://github.com/favicon.ico" width="16" height="16" alt="TenSEAL" /> TenSEAL](https://github.com/OpenMined/TenSEAL), and [<img src="https://huggingface.co/favicon.ico" width="16" height="16" alt="Transformers" /> Hugging Face Transformers](https://huggingface.co/docs/transformers).
Some codes are derived from the [Wanda](https://github.com/locuslab/wanda), [DAGER](https://github.com/insait-institute/dager-gradient-inversion) and [FastMIA](https://github.com/Nikkei/fast-mia) projects, and we thank the authors who have contributed to those great open source works.


