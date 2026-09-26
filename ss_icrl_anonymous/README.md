# SS-ICRL

This repository contains the code required to run Strategy-Structured
In-Context Reinforcement Learning (SS-ICRL) and its ICPO base method.

## Environment

The code is intended for Linux with NVIDIA GPUs and Python 3.11.

```bash
pip install -r requirements.txt
```

The number of GPUs visible through `CUDA_VISIBLE_DEVICES` determines vLLM
tensor parallelism.

## Data

Each task directory must contain `test.json` or `test.parquet`. Each example
must provide the problem in `prompt` (or `problem`) and the reference answer in
`answer` (or `solution`). Benchmark data and model weights are not included.

## Run SS-ICRL

The launcher accepts a Hugging Face model identifier or local model path,
followed by a task directory:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/run_ss_icrl.sh \
  Qwen/Qwen2.5-Math-7B /path/to/AMC-TTT
```

The default method configuration is `configs/ss_icrl.yaml`. It
uses four reasoning strategies, the Reliability-Hill pseudo-label score, and
strategy-specific multi-round context updates.

## Run ICPO

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/run_icpo.sh \
  Qwen/Qwen2.5-Math-7B /path/to/AMC-TTT
```

The ICPO defaults are stored in `configs/icrl.yaml`.

For either launcher, local assets may instead be placed at
`models/<MODEL_NAME>` and `data/<DATASET_NAME>` and referenced by name. Outputs
are written under `results/` by default. Extra command-line arguments are
forwarded to the corresponding Python runner.

## Main files

- `src/icrl/ss_icrl_runner.py`: SS-ICRL and Reliability-Hill scoring.
- `src/icrl/icrl_runner.py`: ICPO runner.
- `src/icrl/multiview_answer_adapter.py`: answer extraction and normalization.
- `scripts/run_ss_icrl.sh`: SS-ICRL launcher.
- `scripts/run_icpo.sh`: ICPO launcher.
