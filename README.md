# E-Gen: Leveraging E-Graphs to Improve Continuous Representations of Symbolic Expressions
# Contrastive Learning

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

Make sure to `copy` the `data` folder under the
[E-Gen](https://github.com/hongbozheng/E-Gen)
repository to this repository.
```
cp /path/to/eeg/data/directory ./
```
- Train data: `data/expr_triplets.txt`
- Test data: `data/exprs_val.txt`

## Train contrastive learning model
#### Train Configuration
To modify training configuration, check `config.py` file.

#### Train
To train transformer encoder with contrastive learning.
```
./train_tx.py
```

## Model
#### Saved model
Saved model will be in `models` folder.

## Test
To test contrastive learning models, checkout `expt` branch.
```
git checkout expt
```
Follow the instructions in `README.md`.
