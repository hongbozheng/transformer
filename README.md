# Embedding Algebra & Embedding K-means Clustering

## Model
#### Trained model
Place trained model(s) in `models` folder.

## Embedding K-means Clustering
#### Inference configuration
To modify inference configuration, check `config.py` file.

#### K-means Clustering Evaluation
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
