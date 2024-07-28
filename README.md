# Semantic Representations of Mathematical Expressions in a Continuous Vector Space

## Dataset
#### Dataset Creation
Clone the [Equivalent Expressions Generation](https://gitlab.engr.illinois.edu/hongboz2/equivalent_expression_generation) repository
```
git clone https://gitlab.engr.illinois.edu/hongboz2/equivalent_expression_generation.git eeg
```
Checkout `dataset` branch
```
git checkout dataset
```
Follow the instructions in `README.md` to properly create the `dataset` and `train`,
`validation`, `test` splits.

Make sure to `copy` the `data` folder under the
[Equivalent Expressions Generation](https://gitlab.engr.illinois.edu/hongboz2/equivalent_expression_generation)
repository to this repository.
```
cp ../path/to/eeg/data/folder ./
```
* Training data: `data/expr_pairs.txt`
* Validation data: `data/exprs_val.txt`

## Training
#### Training Configuration
To modify training configuration, check `config.py` file.

#### Start training
To train seq2seq transformer model
```
./train_tx.py
```

## Model
#### Trained model
Trained model will be in `models` folder
