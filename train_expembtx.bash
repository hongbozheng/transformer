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


# Training Arguments
TRAIN_FILE="data/expr_pairs.txt"
VAL_FILE="data/exprs_val.txt"
SAVE_DIR="models"
PROJECT_NAME="expembtx"
MAX_SEQ_LEN=200
MAX_OUT_LEN=200
MAX_N_POS=200
MAX_TRAIN_EXAMPLES=-1
MAX_VAL_EXAMPLES=-1
SEED=42
TRAIN_BATCH_SIZE=256
VAL_BATCH_SIZE=256
LR=0.0001
WEIGHT_DECAY=0.01
OPTIM="AdamW"
N_EPOCHS=20
TRACK_GRAD_NORM=-1
RUN_NAME="Sep-16"
GRAD_CLIP_VAL=4.0
GRAD_CLIP_ALGO="norm"
PRECISION=32
SYMPY_TIMEOUT=10
EARLY_STOPPING=2
SEMVEC=false
N_MIN_EPOCHS=20
BOOL_DATASET=false
LABEL_SMOOTHING=0.1
AUTOENCODER=false


python3 train_expembtx.py \
    --emb_size $EMB_SIZE \
    --n_heads $N_HEADS \
    --n_encoder_layers $N_ENCODER_LAYERS \
    --n_decoder_layers $N_DECODER_LAYERS \
    --dim_feedforward $DIM_FEEDFORWARD \
    --dropout $DROPOUT \
    --norm_first $NORM_FIRST \
    --activation $ACTIVATION \
    --train_file $TRAIN_FILE \
    --val_file $VAL_FILE \
    --save_dir $SAVE_DIR \
    --project_name $project_name \
    --max_seq_len $MAX_SEQ_LEN \
    --max_out_len $MAX_OUT_LEN \
    --max_n_pos $MAX_N_POS \
    --max_train_examples $MAX_TRAIN_EXAMPLES \
    --max_val_examples $MAX_VAL_EXAMPLES \
    --seed $SEED \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --val_batch_size $VAL_BATCH_SIZE \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --optim $OPTIM \
    --n_epochs $N_EPOCHS \
    --track_grad_norm $TRACK_GRAD_NORM \
    --run_name $RUN_NAME \
    --grad_clip_val $GRAD_CLIP_VAL \
    --grad_clip_algo $GRAD_CLIP_ALGO \
    --precision $PRECISION \
    --sympy_timeout $SYMPY_TIMEOUT \
    --early_stopping $EARLY_STOPPING \
    --semvec $SEMVEC \
    --n_min_epochs $N_MIN_EPOCHS \
    --bool_dataset $BOOL_DATASET \
    --label_smoothing $LABEL_SMOOTHING \
    --autoencoder $AUTOENCODER
