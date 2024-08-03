# Embedding Algebra & Embedding K-means Clustering

## Model
#### Trained model
Place trained model(s) in `models` folder.

## Embedding Algebra
#### Inference configuration
To modify inference configuration, check `config.py` file.

#### Embedding Algebra Evaluation
Check command line input help
```
./emb_algebra.py -h
```

Perform embedding algebra evaluation on quadruple (2-pair) of mathematical expressions
```
./emb_algebra.py -p <pool_filepath> -f <filepath>
```
- `<pool_filepath>` - expression pool filepath
- `<filepath>` - evaluation filepath

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

## Embedding Information Retrieval
#### Inference configuration
To modify inference configuration, check `config.py` file.

#### Information Retrieval Evaluation
Check command line input help
```
./emb_ir.py -h
```

Perform information retrieval evaluation on 1 query and 5 candidates (6 expressions in total)
```
./emb_ir.py -f <filepath>
```
- `<filepath>` - evaluation filepath
