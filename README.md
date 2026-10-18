# CalleeFuse

Code, data, and baseline modifications for the paper:

> **A Callee Fusion-Enhanced Small-Scale Large Model to Alleviate Sub-Method Name Leakage in Method Name Prediction**

## Repository layout

```
.
├── code/                     # CalleeFuse model, training, and evaluation code
│   ├── models/model-2025-10-15/
│   │   ├── T5WithTreeModel.py        # T6ForConditionalGeneration (tree-enhanced CodeT5+)
│   │   ├── test-2025-10-15.py        # training entry point
│   │   ├── generation_task.py        # evaluation / generation entry point
│   │   └── ...
│   ├── gson_reader.py                # JSON dataset reader
│   ├── prepare_datasets.py           # reference data-processing notes
│   └── ...
├── JavaProjectAnalyzer/      # Java method call-tree extractor
├── leakage_analysis.py       # leakage statistics reported in the paper
├── name_length_stats.py      # method-name length distribution table in the paper
├── train_*.json / test_*.json        # preprocessed datasets (8 files)
└── baselines/
    ├── cognac/               # our modifications to the Cognac baseline
    └── gtnm/                 # our modifications to the GTNM baseline
```

## Datasets

Each JSON file is a list of samples; every sample is a recursive call tree:

```json
[
  {
    "root": {
      "method_full_name": "org.pkg.ClassName.methodName(ParamType)",
      "hashCode": "...",
      "method_body": "public void <mask>(...) { ... <extra_id_0>(); }",
      "children": [
        { "method_full_name": "...", "method_body": "...", "children": [] }
      ]
    }
  }
]
```

| File | Purpose |
|---|---|
| `train_d_1_b_128_l_16_no_test.json` / `test_...` | Base data without callee call trees (`d_1`); the Unmasked −ctx baseline rows. |
| `train_d_1_with_callee_body.json` / `test_...` | Callee bodies retained; used by Unmasked +ctx baseline rows. |
| `train_d_2_b_128_l_16_mask_recursive_no_test.json` / `test_...` | Leakage-free data: resolvable callee names are recursively replaced by ordered sentinels (`<extra_id_N>`). |
| `train_d_2_b_128_l_16_random_no_test.json` / `test_...` | Control data: callee names replaced by random method names. |

`b_128` / `l_16` denote the maximum body length (128 tokens) and target-name length (16 tokens).
`no_test` indicates that test methods were removed from the corpus.

## Environment

- Python 3.x, PyTorch, HuggingFace `transformers`, `numpy`, `tqdm`
- Pre-trained checkpoints (not included): CodeT5p-220m, CodeT5, PLBART, CodeGen.
  Place them where the training script expects them (`code/models/model-2025-10-15/test-2025-10-15.py`
  reads `model_path = "../CodeT5p-220m"`), or edit that variable.

## Running CalleeFuse

From the repository root (the JSON files are loaded by relative path):

```bash
# training
PYTHONPATH=code python code/models/model-2025-10-15/test-2025-10-15.py

# evaluation (set root_path / model_path inside the script first)
PYTHONPATH=code python code/models/model-2025-10-15/generation_task.py
```

The dataset, checkpoint directory, and hyperparameters are set as variables at the inside each script.

## Data construction

`JavaProjectAnalyzer/` parses Java projects with JavaParser and emits per-project call-tree JSON files.
It is configured through the constants at the top of `Main.java` (repository
path, JDK path, maximum call-tree depth, minimum number of children).

`code/prepare_datasets.py` is a historical, mostly commented reference for deriving the four
dataset variants: it shows the token-length filtering, `hashCode`-based alignment across variants,
and the final join that produces the `mask_recursive` test file.
It is not a runnable end-to-end pipeline: the intermediate files it reads are not shipped.

## Known issue in the released evaluation code

`T5WithTreeModel.forward` switches the encoder back to training mode after
encoding the callee graph (`self.encoder.train()`). When `generation_task.py`
is run with callee-graph inputs, encoder dropout therefore remains active even
though `model.eval()` was called, and repeated evaluation runs are not
deterministic. If you need deterministic evaluation, save the encoder's mode
before graph encoding and restore it afterwards, e.g.
`was_training = self.encoder.training` / `self.encoder.train(was_training)`.

## Analysis scripts

- `leakage_analysis.py` reproduces the sub-method name leakage statistics.
- `name_length_stats.py` reproduces the method-name length distribution table, in both
  camelCase sub-tokens and CodeT5+ BPE tokens:

  ```bash
  python name_length_stats.py --json train_d_2_b_128_l_16_mask_recursive_no_test.json \
      test_d_2_b_128_l_16_mask_recursive_no_test.json \
      --tokenizer models/CodeT5p-220m --callee --out name_length_stats.json
  ```

  Without `--tokenizer`, only the camelCase measurement is produced (standard library only).

## Baselines

`baselines/cognac/` and `baselines/gtnm/` contain **only the files we added or modified** on top of
the authors' releases of Cognac and GTNM; the upstream code is not redistributed here.

- Both baselines were re-trained on data converted from our JSON files, so their reported numbers
  differ from those in the original publications.
- To reproduce, check out the official release of each baseline and overlay the corresponding
  directory onto it, preserving the relative paths (e.g. `baselines/cognac/training_ptr_gen/train.py`
  replaces `training_ptr_gen/train.py`).
- Each driver runs the ten configurations reported in the paper: four cross settings for the
  sentinel (`mask_recursive`) data, four for the random data, and two unmasked settings.
- `baselines/cognac/COGNAC_EXPERIMENTS.md` and `baselines/gtnm/GTNM_EXPERIMENTS.md` document
  the experiment matrices and run steps.

### Prerequisites for the overlays

The overlay directories are not standalone projects; they must be merged into a full checkout
of the corresponding baseline.

**Cognac**

- Keep the upstream `data_util/` package (`__init__.py`, `batcher.py`, `data.py`, `utils.py`,
  `logging_util.py`) and the remaining `training_ptr_gen/` files (`__init__.py`, `model.py`,
  `eval.py`, `train_util.py`, `decode.py`, `transformer_encoder.py`, ...).
  Our `training_ptr_gen/train.py` replaces the upstream file.
- `data_util/config.py` must exist before the first run: `run_experiments.sh` backs it up and
  rewrites it for every experiment, and the generated configuration is authoritative.
- Copy the eight dataset files used here into `raw_data/`:
  `train/test_d_2_b_128_l_16_random_no_test.json`,
  `train/test_d_2_b_128_l_16_mask_recursive_no_test.json`,
  `train/test_d_1_b_128_l_16_no_test.json`, and
  `train/test_d_1_with_callee_body.json`.
- Install the Ringbo `javalang` fork (`pip install git+https://github.com/Ringbo/javalang.git`);
  the conversion script relies on its `get_token_stream_2` extension.
- Use `environment.yml` (Python 3.7, PyTorch 1.13.1, TensorFlow 2.11, gensim 3.8.3).

**GTNM**

- Keep the upstream `data_processing/` and `model/` packages, in particular
  `data_processing/normalizer.py`, `data_processing/extract_data_subword.py`,
  `data_processing/doc_w2id.txt`, `data_processing/sub_token_w2id.txt`,
  `model/model_invoked.py`, and `model/utils.py`.
- `data_processing/my-languages.so` (a compiled tree-sitter Java grammar) is required by
  `normalizer.py` and is not included here. Build it from the matching tree-sitter-java grammar
  and place it in `data_processing/`:
- Copy the eight dataset files listed under Cognac to the repository root, keeping the names
  shipped here.
- Use `environment.yml` / `Dockerfile` (Python 3.8, TensorFlow 2.10, `sentencepiece`, `pathos`).

## Citation

If you use this code or data, please cite our paper.
```BibTeX
@inproceedings{luo2026calleefuse,
  author    = {Xingyue Luo and Weidong Wang},
  title     = {A Callee Fusion-Enhanced Small-Scale Large Model to Alleviate Sub-Method Name Leakage in Method Name Prediction},
  booktitle = {Proceedings of the 33rd Asia-Pacific Software Engineering Conference (APSEC)},
  year      = {2026},
  note      = {To appear}
}
```

```Word
X. Luo and W. Wang, "A callee fusion-enhanced small-scale large model to alleviate sub-method name leakage in method name prediction," in Proc. 33rd Asia-Pacific Software Engineering Conference (APSEC), 2026, to appear.
```
