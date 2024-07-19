# Contrastive Learning on Mathematical Expression Embeddings

## Train
#### Training Configuration
To modify training configuration, check `config.py` file.

#### Start training
To train transformer model with contrastive learning
```
./train_tx.py
```

## Model
#### Trained model
Trained models will be in `models` folder.


## Evaluation
#### K-Means Clustering
Check command line input help
```
./emb_cluster.py -h
```

Perform K-Means clustering evaluation on groups of equivalent mathematical expressions
```
./emb_cluster.py -f <filepath> -m <method>
```
- `<filepath>` - evaluation filepath
- `<method>` - dimension reduction method
