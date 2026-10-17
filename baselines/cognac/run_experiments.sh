#!/bin/bash
#
# Automated driver for all 10 experiments
# Per experiment: build vocabulary -> train -> decode test
#
# Usage:
#   bash run_experiments.sh          # run all 10 experiments
#   bash run_experiments.sh 1 3 5    # run only experiments 1, 3, 5
#   Experiment 9 = original data (d_1) without masking; callee names are visible in the body
#

export PYTHONPATH=$(pwd)

# Experiment configuration
# Format: "dataset_type train_context test_context"
declare -A EXPERIMENTS
EXPERIMENTS[1]="random with_context with_context"
EXPERIMENTS[2]="random with_context without_context"
EXPERIMENTS[3]="random without_context with_context"
EXPERIMENTS[4]="random without_context without_context"
EXPERIMENTS[5]="mask_recursive with_context with_context"
EXPERIMENTS[6]="mask_recursive with_context without_context"
EXPERIMENTS[7]="mask_recursive without_context with_context"
EXPERIMENTS[8]="mask_recursive without_context without_context"
EXPERIMENTS[9]="original without_context without_context"
# exp10: original data (visible names) + callee-body context for both training and testing
EXPERIMENTS[10]="original_ctx with_context with_context"

ROOT_DIR=$(pwd)
CONFIG_FILE="data_util/config.py"
CONFIG_BACKUP="data_util/config.py.bak"
MAX_ITERS=40000   # default; override with --max-iters N (e.g. exp1/exp4 need 80000)

# Back up the original config
cp "$CONFIG_FILE" "$CONFIG_BACKUP"

# Preprocess the original data (experiment 9): convert the d_1 JSON to Cognac format
preprocess_original() {
    local ds_dir="dataset/original/without_context"
    if [ -f "${ds_dir}/train.json" ] && [ -f "${ds_dir}/test.json" ]; then
        echo "[skip] original data already preprocessed"
        return
    fi
    echo "Preprocessing original data (experiment 9)..."
    python convert_dataset.py --type original --context without || {
        echo "Error: preprocessing the original data failed"
        echo "Check: 1) train_d_1 / test_d_1 exist under raw_data/"
        echo "        2) javalang (Ringbo/javalang fork) is installed"
        return 1
    }
}

# Preprocess exp10 data: d_1_with_callee_body (visible names + completed callee bodies), with_context only
preprocess_original_ctx() {
    local ds_dir="dataset/original_ctx/with_context"
    if [ -f "${ds_dir}/train.json" ] && [ -f "${ds_dir}/test.json" ]; then
        echo "[skip] original_ctx data already preprocessed"
        return
    fi
    echo "Preprocessing original_ctx data (experiment 10)..."
    python convert_dataset.py --type original_ctx --context with || {
        echo "Error: preprocessing original_ctx data failed"
        echo "Check: raw_data/ contains train_d_1_with_callee_body.json / test_d_1_with_callee_body.json"
        return 1
    }
}

# Full preprocessing: regenerate all datasets used by the 10 experiments from raw_data
# For a "full rerun": clear the old dataset products (keeping the vocabulary dirs), then reconvert from scratch with javalang.
preprocess_all() {
    echo "============================================"
    echo "[full preprocessing] Regenerating all datasets from raw_data"
    echo "============================================"
# Remove old datasets (only conversion products; keep exp*_fasttext_vectors, which train_fasttext_exp.py rebuilds)
    rm -rf dataset/random dataset/mask_recursive dataset/original dataset/original_ctx
    python convert_dataset.py --generate-all || {
        echo "Error: full dataset preprocessing failed"
        echo "Check: 1) the six source files train_d_1/d_2 and test_d_1/d_2 exist under raw_data/"
        echo "        2) javalang (Ringbo/javalang fork) is installed"
        return 1
    }
    python convert_dataset.py --type original_ctx --context with || {
        echo "Error: original_ctx preprocessing failed"
        echo "Check: raw_data/ contains train_d_1_with_callee_body.json / test_d_1_with_callee_body.json"
        return 1
    }
    echo "[full preprocessing] Done"
    echo ""
}

# Full vocabulary rebuild: rebuild the FastText vocabulary of each of the 10 experiments
build_all_vocab() {
    echo "============================================"
    echo "[full vocab build] Rebuilding FastText vocabularies for the 10 experiments"
    echo "============================================"
    rm -rf dataset/exp*_fasttext_vectors
    python train_fasttext_exp.py --all || {
        echo "Error: full vocabulary build failed"
        return 1
    }
    echo "[full vocab build] Done"
    echo ""
}

run_experiment() {
    local exp_id=$1
    local config="${EXPERIMENTS[$exp_id]}"
    local ds=$(echo $config | cut -d' ' -f1)
    local train_ctx=$(echo $config | cut -d' ' -f2)
    local test_ctx=$(echo $config | cut -d' ' -f3)

    echo "============================================"
    echo "Experiment $exp_id: $ds / train=$train_ctx / test=$test_ctx"
    echo "============================================"

# 1. Build the vocabulary (skip if present; full mode rebuilds it in build_all_vocab)
    echo "[step 1/3] Building vocabulary..."
    if [ -f "dataset/exp${exp_id}_fasttext_vectors/vocab.pkl" ]; then
        echo "[skip] vocabulary for experiment ${exp_id} already exists"
    else
        python train_fasttext_exp.py --exp $exp_id
    fi

# 2. Set the paths in config.py
    local train_path="dataset/${ds}/${train_ctx}/train.json"
    local eval_path="dataset/${ds}/${train_ctx}/test.json"
    local decode_path="dataset/${ds}/${test_ctx}/test.json"
    local vocab_path="dataset/exp${exp_id}_fasttext_vectors/vocab.pkl"

# Generate config.py with Python to avoid sed matching issues
    cp "$CONFIG_BACKUP" "$CONFIG_FILE"
    python -c "
import os
config_content = '''import os

root_dir = \"${ROOT_DIR}/\"

train_data_path = os.path.join(root_dir, \"${train_path}\")
eval_data_path = os.path.join(root_dir, \"${eval_path}\")
decode_data_path = os.path.join(root_dir, \"${decode_path}\")
vocab_path = os.path.join(root_dir, \"${vocab_path}\")
log_root = os.path.join(root_dir, \"log\")
excluded_type = {}

# Hyperparameters
hidden_dim= 400
emb_dim= 150

stmt_emb_dim = 20
batch_size = 120
max_enc_steps=350
max_dec_steps=6
beam_size=4
min_dec_steps=1
vocab_size=85000
stmt_size = 40
lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0
dropout_prob = 0.25
pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = ${MAX_ITERS}

use_gpu=True

lr_coverage=0.15

log_interval = 1000
eval_interval = 2000
console_log_level = \"INFO\"
file_log_level = \"DEBUG\"
'''
with open('${CONFIG_FILE}', 'w') as f:
    f.write(config_content)
"

# 3. Train
    echo "[step 2/3] Training the model..."
    python training_ptr_gen/train.py 2>&1 | tee "log/exp${exp_id}_train.log"

# Find the newest training directory
    local model_dir=$(ls -td log/train_* 2>/dev/null | head -1)
    if [ -z "$model_dir" ]; then
        echo "Error: training output directory not found"
        return 1
    fi

# 4. Two-stage evaluation (aligned with GTNM; prevents cross experiments from selecting checkpoints on the target test set)
#    (a) evaluate all checkpoints on the [training-side validation set eval_path] and select the best by F1;
#    (b) test that fixed checkpoint on the [target test set decode_path] as the final result.
#    Self-test experiments have eval_path==decode_path (stages equivalent); for cross experiments checkpoint selection sees no target-set information, so there is no leakage.
    echo "[step 3/3] Evaluating all checkpoints (selecting the best on the training-side validation set)..."
    mkdir -p "results"

    # (a) evaluate all checkpoints on the training-side validation set for checkpoint selection
    python evaluate.py "${model_dir}/model/" --decode-data "${eval_path}" \
        --output "results/exp${exp_id}_valselect.json" 2>&1 | tee "log/exp${exp_id}_eval.log"

    # Select the best checkpoint by training-side validation F1
    local best_model=$(python -c "
import json
with open('results/exp${exp_id}_valselect.json', 'r') as f:
    results = json.load(f)
if isinstance(results, dict):
    results = [results]
best = max(results, key=lambda x: x['overall']['f1'])
print(best['model'])
")
    echo "Best checkpoint on the training-side validation set: $best_model"

    if [ -n "$best_model" ] && [ -f "$best_model" ]; then
        # (b) test the selected checkpoint on the target test set for the final result
        echo "Testing the best checkpoint on the target test set: ${decode_path}"
        python evaluate.py "$best_model" --decode-data "${decode_path}" \
            --output "results/exp${exp_id}.json" 2>&1 | tee -a "log/exp${exp_id}_eval.log"

        # Keep the convergence curve: store the per-checkpoint training-side results as _all.json (for extract_convergence.py)
        cp "results/exp${exp_id}_valselect.json" "results/exp${exp_id}_all.json"

        # Copy the best model to best_model
        cp "$best_model" "${model_dir}/model/best_model"
        echo "Best model: $best_model -> best_model"

        # Remove the other checkpoints
        for f in ${model_dir}/model/model_*; do
            if [ "$f" != "$best_model" ]; then
                rm -f "$f"
            fi
        done
        # Remove the original file of the best model (already copied to best_model)
        rm -f "$best_model"
    fi

    # Rename the training directory to identify the experiment
    mv "$model_dir" "log/exp${exp_id}_${ds}_train${train_ctx}_test${test_ctx}"

    echo "Experiment $exp_id finished"
    echo ""
}

# Parse arguments:
#   --fresh        full rerun (regenerate datasets from raw_data + rebuild all vocabularies)
#   --max-iters N  override max_iterations (default 40000; exp1/exp4 use 80000)
FRESH=0
POSITIONAL=""
while [ $# -gt 0 ]; do
    case "$1" in
        --fresh)
            FRESH=1
            shift
            ;;
        --max-iters)
            MAX_ITERS="$2"
            shift 2
            ;;
        *)
            POSITIONAL="$POSITIONAL $1"
            shift
            ;;
    esac
done
set -- $POSITIONAL

# Determine the experiments to run
if [ $# -eq 0 ]; then
    exp_list="1 2 3 4 5 6 7 8 9"
else
    exp_list="$@"
fi

if [ "$FRESH" == "1" ]; then
    # Full rerun: regenerate all datasets from raw_data + rebuild all vocabularies
    preprocess_all || exit 1
    build_all_vocab || exit 1
else
    # Incremental mode: preprocess the exp9/10 source data only when missing
    if [[ " $exp_list " == *" 9 "* ]]; then
        preprocess_original
    fi
    if [[ " $exp_list " == *" 10 "* ]]; then
        preprocess_original_ctx
    fi
fi

echo "Will run: $exp_list"
echo ""

for exp_id in $exp_list; do
    run_experiment $exp_id
done

# Restore config
cp "$CONFIG_BACKUP" "$CONFIG_FILE"
echo "All experiments finished. config.py restored."
