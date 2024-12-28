# E-Gen: Leveraging E-Graphs to Improve Continuous Representations of Symbolic Expressions
# Seq2seq Transformer Model

## Dataset
#### Dataset Creation
Clone the [E-Gen](https://github.com/hongbozheng/E-Gen) repository.
```
git clone git@github.com:hongbozheng/E-Gen.git e-gen
```
Checkout `dataset` branch.
```
git checkout dataset
```
Follow the instructions in `README.md` to properly create the `dataset` and
`train` & `test` splits.

Make sure to `copy` the `data` directory under the
[E-Gen](https://github.com/hongbozheng/E-Gen)
repository to this repository.
```
cp /path/to/eeg/data/directory ./
```
- Train data: `data/expr_pairs.txt`
- Test data: `data/exprs_val.txt`

## Train seq2seq transformer model
#### Train Configuration
To modify train configuration, check `config.py` file.

#### Train
To train seq2seq transformer model.
```
./train_tx.py
```
To train transformer encoder with contrastive learning,
checkout the `cl` branch.
```
git checkout cl
```
Follow the instructions in `README.md`.

## Model
#### Saved model
Saved seq2seq transformer model will be in `models` directory.

## Test
#### Test Configuration
To modify test configuration, check `config.py` file.

#### Test
To test seq2seq transformer model.
```
./test_tx.py
```
