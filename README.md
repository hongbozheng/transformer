# Semantic Representations of Mathematical Expressions in a Continuous Vector Space

## Environment Setup

Setup the environment using `conda` as follows:
```
conda env create -n expembtx -f environment_new.yml
```

## Dataset
#### Dataset Creation
##### Equivalent Expressions Generation (EEG)
Clone the [Equivalent Expressions Generation](https://gitlab.engr.illinois.edu/hongboz2/equivalent_expression_generation) repository
```
git clone https://gitlab.engr.illinois.edu/hongboz2/equivalent_expression_generation.git eeg
```
Checkout `dataset` branch
```
git checkout dataset
```
All the generated equivalent expressions are in the `equiv_exprs` folder

Follow the instructions in section `Dataset(Python)` in `README.md` to properly create the `dataset` and `train`,
`validation`, `test` splits.

Make sure to `copy` the `data` folder in
[Equivalent Expressions Generation](https://gitlab.engr.illinois.edu/hongboz2/equivalent_expression_generation)
repository to this repository.
```
cp ../path/to/eeg/data/folder ./
```
* Training data: `data/expr_pairs_train.txt`
* Validation data: `data/expr_pairs_val.txt`
* Testing data: `data/expr_pairs_test.txt`

## Model
Trained model will be in `models` folder

## Training
#### Training
To train ExpEmb on the Equivalent Expressions Dataset, `train_expembtx.py` may be used.

Example:
```
python train_expembtx.py \
    --train_file <TRAIN_FILE> \
    --val_file <VAL_FILE> \
    --n_epochs <N_EPOCHS> \
    --norm_first True \
    --optim Adam \
    --weight_decay 0 \
    --lr 0.0001 \
    --train_batch_size <TRAIN_BATCH_SIZE> \
    --run_name <RUN_NAME> \
    --val_batch_size <EVAL_BATCH_SIZE> \
    --grad_clip_val 1 \
    --max_out_len 256 \
    --precision 16 \
    --save_dir <OUT_DIR> \
    --early_stopping <EARLY_STOPPING> \
    --n_min_epochs <N_MIN_EPOCHS> \
    --label_smoothing 0.1 \
    --seed 42
```

For all supported options, use `python train_expembtx.py --help` or refer to [TrainingAgruments](expemb/args.py#TrainingAgruments).

## Evaluation
To evaluate a trained model, `test_expembtx.py` may be used. 

For the Equivalent Expressions Dataset, the following command may be used to test the model accuracy. On completion, it will generate a file containing the results inside save_dir:'models/equivexp/' with `<RESULT_FILE_PREFIX>` as the file name prefix.
```
python test_expembtx.py \
    --test_file 'data/test.test' \
    --save_dir 'models/equivexp/' \
    --beam_sizes 1 10 50 \
    --max_seq_len 256 \
    --result_file_prefix <RESULT_FILE_PREFIX> \
    --batch_size 32
```

For all supported options, use `python test_expembtx.py --help` or refer to [TestingArguments](expemb/args.py#TestingArguments).