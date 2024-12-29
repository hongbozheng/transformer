# E-Gen: Leveraging E-Graphs to Improve Continuous Representations of Symbolic Expressions

## Expriments

## Model
### Saved model
Saved model(s) should be in `models` directory.

## Configuration
To modify configuration, check `config.py` file.

## K-Means Clustering
K-Means clustering evaluation on groups of equivalent mathematical expressions.

Check command line input help.
```
./k-means.py -h
```
Perform evaluation.
```
./k-means.py -m <ckpt> -e <emb mode> -f <filepath> -d <dim red>
```
- `<ckpt>` - model checkpoint filepath
- `<emb mode>` - embedding mode
  - mean - average pooling
  - max - max pooling
- `<filepath>` - test filepath
- `<dim red> (optional)` - dimensionality reduction method
  - t-SNE
  - UMAP

## Formula Selection
Select the equivalent expression from a group of 7 expressions
(1 equivalent and 6 distractors). 1 equivalent to query,
3 in-equivalent but syntactically similar to the query, and 
3 in-equivalent but syntactically similar to the correct answer.

Check command line input help.
```
./formula_select.py -h
```
Perform evaluation.
```
./formula_select.py -m <ckpt> -e <emb mode> -k <top-k> -f <filepath>
```
- `<ckpt>` - model checkpoint filepath
- `<emb mode>` - embedding mode
  - mean - average pooling
  - max - max pooling
- `<top-k>` - top-k expressions to select
- `<filepath>` - test filepath

## Mistake Detection
Find mistake(s) from a sequence of mathematical derivations.

Check command line input help.
```
./mistake_detect.py -h
```
Perform evaluation.
```
./mistake_detect.py -m <ckpt> -e <emb mode> -f <filepath>
```
- `<ckpt>` - model checkpoint filepath
- `<emb mode>` - embedding mode
  - mean - average pooling
  - max - max pooling
- `<filepath>` - test filepath

## Embedding Algebra
Select the best expression from a pool of expressions based on a given pair of 
mathematical expressions and a given expression.

Check command line input help.
```
./emb_algebra.py -h
```
Perform evaluation.
```
./emb_algebra.py -m <ckpt> -e <emb mode> -p <pool> -f <filepath>
```
- `<ckpt>` - model checkpoint filepath
- `<emb mode>` - embedding mode
  - mean - average pooling
  - max - max pooling
- `<pool>` - expression pool to select from
- `<filepath>` - test filepath
