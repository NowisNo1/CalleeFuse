# Cognac experiments

This directory contains only the files we added or replaced on the upstream Cognac release.
See the repository README for upstream prerequisites, environment, and the data files.

## Files

| Path | Purpose |
|---|---|
| `convert_dataset.py` | Converts the JSON datasets to Cognac input format with javalang (Ringbo fork required). Supports `random`, `mask_recursive`, `original`, and `original_ctx`. |
| `train_fasttext_exp.py` | Builds one FastText vocabulary per experiment. |
| `evaluate.py` | Evaluates checkpoints and writes overall and per-target-length metrics. |
| `extract_convergence.py` | Extracts convergence curves from `results/exp*_all.json`. |
| `run_experiments.sh` | Driver: data conversion (if needed), vocabulary, training, and two-stage evaluation. |
| `training_ptr_gen/train.py` | Replaces the upstream training entry point. |
| `environment.yml` | Conda environment. |

## Experiment matrix

`with context` adds the callee's return type, parameter types, and body tokens to the input;
`without context` uses only the target method (class name, return type, parameters, body).

| ID | Data | Train | Test | Paper row |
|---|---|---|---|---|
| 1 | `random` | with | with | Random +ctx → +ctx |
| 2 | `random` | with | without | Random +ctx → −ctx |
| 3 | `random` | without | with | Random −ctx → +ctx |
| 4 | `random` | without | without | Random −ctx → −ctx |
| 5 | `mask_recursive` | with | with | Sentinel +ctx → +ctx |
| 6 | `mask_recursive` | with | without | Sentinel +ctx → −ctx |
| 7 | `mask_recursive` | without | with | Sentinel −ctx → +ctx |
| 8 | `mask_recursive` | without | without | Sentinel −ctx → −ctx |
| 9 | `original` | without | without | Unmasked −ctx |
| 10 | `original_ctx` | with | with | Unmasked +ctx |

Data variants:

- `mask_recursive` = Sentinel: callee names replaced by `<extra_id_N>`.
- `random` = Random: callee names replaced by random method names.
- `original` = Unmasked −ctx: `d_1` data with visible names and no callee bodies.
- `original_ctx` = Unmasked +ctx: `d_1_with_callee_body` data with visible names and callee bodies.

## Run

Place the eight dataset JSON files in `raw_data/`, then:

```bash
bash run_experiments.sh              # all 10 experiments
bash run_experiments.sh 5            # experiment 5 only
bash run_experiments.sh 1 4          # experiments 1 and 4
bash run_experiments.sh --fresh      # regenerate all converted data and vocabularies
```

`run_experiments.sh` writes `data_util/config.py` for each experiment from an embedded template
(the upstream file is backed up as `data_util/config.py.bak`). Checkpoints are selected on the
training-side validation set (`eval_path`) by F1, then the selected checkpoint is evaluated on
the target test set (`decode_path`), so cross experiments do not select on the target test set.

## Outputs

- `dataset/{data}/{with_context,without_context}/{train,test}.json` — converted data.
- `dataset/exp<ID>_fasttext_vectors/` — per-experiment vocabularies.
- `results/exp<ID>.json` — overall P/R/F1/EM and the per-length breakdown.
- `results/exp<ID>_{valselect,all}.json` — per-checkpoint validation results.
- `log/exp<ID>_{train,eval}.log` — training and evaluation logs.

Metrics are computed on camelCase sub-tokens, consistent with Cognac's original evaluation.
