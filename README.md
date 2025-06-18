# E-Gen: Leveraging E-Graphs to Improve Continuous Representations of Symbolic Expressions

## Seq2seq Transformer Model

## Dataset
### Dataset Creation
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
cp /path/to/e-gen/data/directory ./
```
- Train data: `data/equiv_pair.txt`
- Test data: `data/val.txt`

## Train seq2seq transformer model
### Train Configuration
- To modify train configuration, check `cfgs/models/seq2seq.yaml` file
- To modify dataset configuration, check `cfgs/datasets/equiv_pair.yaml` file

### Train
To train seq2seq transformer model
```
./train_model.py --cfgs cfgs/models/seq2seq.yaml --dataset cfgs/datasets/equiv_pair.yaml
```

## Train encoder with contrastive learning
### Train Configuration
- To modify train configuration, check `cfgs/models/math_enc.yaml` file
- To modify dataset configuration, check `cfgs/datasets/contrastive_expr.yaml` file

### Train
To train encoder model
```
./train_model.py --cfgs cfgs/models/math_enc.yaml --dataset cfgs/datasets/contrastive_expr.yaml
```

## Test seq2seq transformer model
To test seq2seq transformer model
```
./test_model.py --cfgs cfgs/models/seq2seq.yaml --dataset cfgs/datasets/equiv_pair.yaml
```
