## Discrete Representation of Time Series Data using VAE with Discrete Latent Variables

### Summary


### Todos 

Priority:
- model encode per patch
- replicate plot from presentation
- enc, dec: 1d convolution, lstm
- implement 1 downstream task
- see if similar patches result in similar binary codes
- baseline -> cluster binary codes , time series patches

Nice to have

- multivariate bernoulli??
- variable #bits and #symbols
- piecewiese linear methods (e.g. PAA)
- normalization -> time series scale
- additional "magnitude"+"shift" word? -> dependent on downstream tasks
- ensemble method to encode differnet attributes using differnt vaes
- generalization to other datasets
- Experiment management (wandb, mlflow)