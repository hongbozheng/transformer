#!/bin/bash


export WANDB_MODE=disabled


# Model Arguments
EMB_SIZE=512
N_HEADS=8
N_ENCODER_LAYERS=6
N_DECODER_LAYERS=6
DIM_FEEDFORWARD=2048
DROPOUT=0.1
NORM_FIRST=true
ACTIVATION="relu"


# Testing Arguments
TEST_FILE="data/expr_pairs_test.txt"
SAVE_DIR="models"
FULL_FILE=""
BEAM_SIZES=1
SYMPY_TIMEOUT=10
BATCH_SIZE=1024
CKPT_NAME="best"
MAX_SEQ_LEN=512
MAX_TEST_EXAMPLES=-1
PRECISION=32
SEMVEC=false
RESULT_FILE_PREFIX="results"


python3 test_expembtx.py \
    --test_file $TEST_FILE \
    --save_dir $SAVE_DIR \
    --full_file $FULL_FILE \
    --beam_sizes $BEAM_SIZES \
    --sympy_timeout $SYMPY_TIMEOUT \
    --batch_size $BATCH_SIZE \
    --ckpt_name $CKPT_NAME \
    --max_seq_len $MAX_SEQ_LEN \
    --max_test_examples $MAX_TEST_EXAMPLES \
    --precision $PRECISION \
    --semvec $SEMVEC \
    --result_file_prefix $RESULT_FILE_PREFIX