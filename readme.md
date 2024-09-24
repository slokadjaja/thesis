## Symbolic Representation of Time Series Data using VAEs with Discrete Latent Variables

### Summary

We aim to learn symbolic representations of time series data in an unsupervised manner using 
variational autoencoders (VAEs) with categorical latent variables + Gumbel-softmax reparametrization

MLFlow is used to log metrics, parameters and artefacts. To view runs in UI, run: ```mlflow server --host 127.0.0.1 --port 8080 ```

Model is tested using two downstream tasks:

- Time series classification using UCR dataset
- Portfolio hedging
    <details>
    
    Downstream task as described in 'Stock Embeddings: Representation Learning for Financial Time Series' and
    'Contrastive Learning of Asset Embeddings from Financial Time Series'
    
    To reduce investment risk, portfolio managers use diversification and hedging, measuring effectiveness in terms of
    volatility reduction. As a result, identifying dissimilar stocks that behave oppositely to similar ones is essential
    for traders to hedge their target stocks and limit overall risk.
    
    Typically, hedging involves negatively correlated assets and various correlation metrics. We propose an alternative:
    using generated embeddings to find maximally dissimilar stocks and inform hedging strategies. We evaluate a scenario
    where an investor holds a position in a stock (query stock) and seeks a single stock (hedge stock) to reduce risk,
    measured as volatility, as much as possible.
    
    We test a hedging approach by using two-asset long portfolio, consisting of an anchor asset, and the other asset having
    the lowest similarity in the latent space, measured using hamming distance. Embeddings will be computed using train
    horizon, and portfolio simulated on out-of-sample horizon.
    
    Benchmark: pearson correlation of returns
    
    </details>