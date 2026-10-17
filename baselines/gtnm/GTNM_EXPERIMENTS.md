# GTNM experiments

This directory contains only the files we added or replaced on the upstream GTNM release.
See the repository README for upstream prerequisites (including `data_processing/my-languages.so`)
and the data files.

## Files

| Path | Purpose |
|---|---|
| `data_processing/convert_json_data.py` | Converts the JSON datasets to GTNM pickle format. Uses the upstream tree-sitter Normalizer; `--use_children_context` builds the project-context tensors. |
| `data_processing/doc_w2id.txt`, `data_processing/sub_token_w2id.txt` | Document and code sub-token vocabularies. |
| `model/hparams.py` | Hyperparameters; data and output paths can be overridden on the command line. |
| `model/train.py`, `model/test.py` | Training and testing entry points with F1 early stopping (patience 5, `min_delta` 0.001). |
| `run_all.sh` | Driver: preprocessing, training 6 models, testing 10 train/test pairs. Resumable. |
| `Dockerfile`, `environment.yml` | Environment. |

## Data

Place the eight dataset JSON files at the repository root. Variants:

- `mask_recursive` = Sentinel: callee names replaced by `<extra_id_N>`.
- `random` = Random: callee names replaced by random method names.
- `original` = Unmasked −ctx: `d_1` data with visible names and no callee bodies.
- `original_ctx` = Unmasked +ctx: `d_1_with_callee_body` data with visible names and callee bodies.

`with context` means the recursively collected callee bodies are encoded as project context
(`pro`); `without context` means all-zero project context.

## Experiment matrix

| ID | Model (training data) | `pro` | Test set | Paper row |
|---|---|---|---|---|
| 1 | `random_no_ctx` | False | `random_no_ctx` | Random −ctx → −ctx |
| 2 | `random_no_ctx` | False | `random_with_ctx` | Random −ctx → +ctx† |
| 3 | `random_with_ctx` | True | `random_no_ctx` | Random +ctx → −ctx |
| 4 | `random_with_ctx` | True | `random_with_ctx` | Random +ctx → +ctx |
| 5 | `mask_no_ctx` | False | `mask_no_ctx` | Sentinel −ctx → −ctx |
| 6 | `mask_no_ctx` | False | `mask_with_ctx` | Sentinel −ctx → +ctx† |
| 7 | `mask_with_ctx` | True | `mask_no_ctx` | Sentinel +ctx → −ctx |
| 8 | `mask_with_ctx` | True | `mask_with_ctx` | Sentinel +ctx → +ctx |
| 9 | `original` | False | `original` | Unmasked −ctx |
| 10 | `original_ctx` | True | `original_ctx` | Unmasked +ctx |

† When `--pro` is False, the context encoder is not part of the model graph, so test-time
context is unavailable and the two test sets give identical results.

## Run

```bash
bash run_all.sh
```

The driver performs all conversion, training, and testing steps and skips completed ones on
rerun. To run individual commands, use the same invocations shown inside `run_all.sh`.

## Outputs

- `data_processed/java-large/` — converted pickle data.
- `saved/java-large/<model>/ckpt/`, `log/`, `res.txt` — checkpoints, logs, and training results.
- `saved/java-large/test_results_<model>__on__<test-set>.txt` — final test results.

Metrics are overall precision, recall, F1, and exact match, as computed by `model/test.py`.
