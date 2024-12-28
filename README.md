# E-Gen: Leveraging E-Graphs to Improve Continuous Representations of Symbolic Expressions
# seq2seq Model

## Dataset
#### Dataset Creation
Clone the [E-Gen](https://github.com/hongbozheng/E-Gen) repository
```
git clone git@github.com:hongbozheng/E-Gen.git e-gen
```
Checkout `dataset` branch
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
- Train data: `data/expr_pairs.txt`
- Test data: `data/exprs_val.txt`

## Training
#### Training Configuration
To modify training configuration, check `config.py` file.

#### Start training
To train seq2seq transformer model
```
./train_tx.py
```
To train transformer encoder with contrastive learning,
checkout the `cl` branch
```
git checkout cl
```
Follow the instructions in `README.md`

## Model
#### Trained model
Trained model will be in `models` folder.

## Testing
#### Testing Configuration
To modify testing configuration, check `config.py` file.
