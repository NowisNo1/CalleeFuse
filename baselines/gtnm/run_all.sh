#!/bin/bash
# GTNM full pipeline: preprocessing -> training -> testing
# Run inside the Docker container: bash run_all.sh
# Resumable: rerunning after an interruption skips completed steps
set -e

DATA_DIR="data_processed/java-large"
SAVE_DIR="saved/java-large"

mkdir -p "$DATA_DIR"

echo "=========================================="
echo "Stage 1: data preprocessing"
echo "=========================================="

# Preprocess helper: skip when the output files already exist
preprocess() {
    local INPUT=$1
    local OUTPUT_PREFIX=$2
    local EXTRA_ARGS=$3

    if [ -f "${OUTPUT_PREFIX}_body.pkl" ] && [ -f "${OUTPUT_PREFIX}_tag.pkl" ]; then
        echo "[skip] ${OUTPUT_PREFIX} already exists"
        return
    fi

    echo "[process] $INPUT -> $OUTPUT_PREFIX"
    python3 data_processing/convert_json_data.py \
        --input_file "$INPUT" \
        --output_prefix "$OUTPUT_PREFIX" \
        $EXTRA_ARGS
}

# training sets
preprocess train_d_2_b_128_l_16_random_no_test.json \
    ${DATA_DIR}/train_random_no_ctx_subword

preprocess train_d_2_b_128_l_16_random_no_test.json \
    ${DATA_DIR}/train_random_with_ctx_subword \
    "--use_children_context"

preprocess train_d_2_b_128_l_16_mask_recursive_no_test.json \
    ${DATA_DIR}/train_mask_no_ctx_subword

preprocess train_d_2_b_128_l_16_mask_recursive_no_test.json \
    ${DATA_DIR}/train_mask_with_ctx_subword \
    "--use_children_context"

# test sets
preprocess test_d_2_b_128_l_16_random_no_test.json \
    ${DATA_DIR}/test_random_no_ctx_subword

preprocess test_d_2_b_128_l_16_random_no_test.json \
    ${DATA_DIR}/test_random_with_ctx_subword \
    "--use_children_context"

preprocess test_d_2_b_128_l_16_mask_recursive_no_test.json \
    ${DATA_DIR}/test_mask_no_ctx_subword

preprocess test_d_2_b_128_l_16_mask_recursive_no_test.json \
    ${DATA_DIR}/test_mask_with_ctx_subword \
    "--use_children_context"

# original data (experiment 9): d_1 without context
preprocess train_d_1_b_128_l_16_no_test.json \
    ${DATA_DIR}/train_original_subword

preprocess test_d_1_b_128_l_16_no_test.json \
    ${DATA_DIR}/test_original_subword

# original data + context (experiment 10): d_1_with_callee_body (visible names + completed callee bodies), with context
preprocess train_d_1_with_callee_body.json \
    ${DATA_DIR}/train_original_ctx_subword \
    "--use_children_context"

preprocess test_d_1_with_callee_body.json \
    ${DATA_DIR}/test_original_ctx_subword \
    "--use_children_context"

echo "Preprocessing finished!"

echo "=========================================="
echo "Stage 2: training 4 models"
echo "=========================================="

train_model() {
    local EXP_NAME=$1
    local TRAIN_PREFIX=$2
    local EVAL_PREFIX=$3
    local USE_PRO=$4

    local MODEL_DIR="${SAVE_DIR}/${EXP_NAME}/"

    # Skip training when a checkpoint already exists
    if [ -f "${MODEL_DIR}/ckpt/checkpoint" ]; then
        echo "[skip] $EXP_NAME already has a checkpoint; skipping training"
        return
    fi

    echo "------------------------------------------"
    echo "Training model: $EXP_NAME (pro=$USE_PRO)"
    echo "  training data: $TRAIN_PREFIX"
    echo "  validation data: $EVAL_PREFIX"
    echo "------------------------------------------"

    mkdir -p "${MODEL_DIR}/ckpt" "${MODEL_DIR}/log"

    # Create symlinks: train_subword_*.pkl -> actual training data
    for suffix in body pro doc tag invoked; do
        ln -sf "$(pwd)/${DATA_DIR}/${TRAIN_PREFIX}_${suffix}.pkl" "${DATA_DIR}/train_subword_${suffix}.pkl"
        ln -sf "$(pwd)/${DATA_DIR}/${EVAL_PREFIX}_${suffix}.pkl" "${DATA_DIR}/eval_subword_${suffix}.pkl"
    done

    python3 -m model.train --gpu 0 --pro $USE_PRO \
        --data_path "${DATA_DIR}/" \
        --evaldir "${DATA_DIR}/" \
        --ckpt_dir "${MODEL_DIR}/ckpt/" \
        --logdir "${MODEL_DIR}/log/" \
        --res_log "${MODEL_DIR}/res.txt"

    echo "Model $EXP_NAME trained and saved to $MODEL_DIR"
}

# Random group
train_model "random_no_ctx"   "train_random_no_ctx_subword"   "test_random_no_ctx_subword"   False
train_model "random_with_ctx" "train_random_with_ctx_subword" "test_random_with_ctx_subword" True

# Mask group
train_model "mask_no_ctx"   "train_mask_no_ctx_subword"   "test_mask_no_ctx_subword"   False
train_model "mask_with_ctx" "train_mask_with_ctx_subword" "test_mask_with_ctx_subword" True

# Original group (experiment 9)
train_model "original"   "train_original_subword"   "test_original_subword"   False

# Original + context group (experiment 10): visible names + callee-body context, with pro enabled
train_model "original_ctx" "train_original_ctx_subword" "test_original_ctx_subword" True

echo "All training finished!"

echo "=========================================="
echo "Stage 3: testing (each model on 2 test sets)"
echo "=========================================="

test_model() {
    local EXP_NAME=$1
    local TEST_PREFIX=$2
    local RESULT_TAG=$3
    local USE_PRO=$4

    local RESULT_FILE="${SAVE_DIR}/test_results_${RESULT_TAG}.txt"

    # Skip when the result file already exists
    if [ -f "$RESULT_FILE" ] && [ -s "$RESULT_FILE" ]; then
        echo "[skip] result $RESULT_TAG already exists"
        return
    fi

    echo "------------------------------------------"
    echo "Test: model=$EXP_NAME, test set=$TEST_PREFIX (pro=$USE_PRO)"
    echo "------------------------------------------"

    local MODEL_DIR="${SAVE_DIR}/${EXP_NAME}/"

    # Create symlinks: test_subword_*.pkl -> actual test data
    for suffix in body pro doc tag invoked; do
        ln -sf "$(pwd)/${DATA_DIR}/${TEST_PREFIX}_${suffix}.pkl" "${DATA_DIR}/test_subword_${suffix}.pkl"
    done

    python3 -m model.test --gpu 0 --pro $USE_PRO \
        --data_path "${DATA_DIR}/" \
        --testdir "${DATA_DIR}/" \
        --ckpt_dir "${MODEL_DIR}/ckpt/" \
        --logdir "${MODEL_DIR}/log/" \
        --res_log "$RESULT_FILE"

    echo "Writing results to: $RESULT_FILE"
}

# Random group
test_model "random_no_ctx"   "test_random_no_ctx_subword"   "random_no_ctx__on__random_no_ctx"       False
test_model "random_no_ctx"   "test_random_with_ctx_subword" "random_no_ctx__on__random_with_ctx"     False
test_model "random_with_ctx" "test_random_no_ctx_subword"   "random_with_ctx__on__random_no_ctx"     True
test_model "random_with_ctx" "test_random_with_ctx_subword" "random_with_ctx__on__random_with_ctx"   True

# Mask group
test_model "mask_no_ctx"   "test_mask_no_ctx_subword"   "mask_no_ctx__on__mask_no_ctx"       False
test_model "mask_no_ctx"   "test_mask_with_ctx_subword" "mask_no_ctx__on__mask_with_ctx"     False
test_model "mask_with_ctx" "test_mask_no_ctx_subword"   "mask_with_ctx__on__mask_no_ctx"     True
test_model "mask_with_ctx" "test_mask_with_ctx_subword" "mask_with_ctx__on__mask_with_ctx"   True

# Original group (experiment 9)
test_model "original"   "test_original_subword"   "original__on__original"   False

# Original + context group (experiment 10): self-test, pro=True
test_model "original_ctx" "test_original_ctx_subword" "original_ctx__on__original_ctx" True

echo "=========================================="
echo "All experiments finished!"
echo "Result files: ${SAVE_DIR}/test_results_*.txt"
echo "=========================================="
