## Symbolic Representation of Time Series Data using VAEs with Discrete Latent Variables

### Summary
We aim to learn symbolic representations of time series data in an unsupervised manner using 
variational autoencoders (VAEs) with categorical latent variables + Gumbel-softmax reparametrization

This project was developed using Python 3.11.9, package requirements are listed in ```requirements.txt```

### Experiment tracking
MLFlow is used to log metrics, parameters and artefacts. To view runs in UI, run: ```mlflow server --host 127.0.0.1 --port 8080 ```.
If can't connect due to connection in use, run ```pkill -f gunicorn```

It is also possible to log MLFlow runs in Azure ML. To do this, create an ```.env``` file in the root directory, and store
the environment variables ```AZURE_TENANT_ID```, ```AZURE_CLIENT_ID```, ```AZURE_CLIENT_SECRET``` of your service principal, and 
the ```TRACKING_URI``` that points to your Azure ML workspace.

To query runs in Azure ML using python SDK and get metrics and artefacts, 
refer to [microsoft learn](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-track-experiments-mlflow?view=azureml-api-2&source=recommendations)

### Downstream tasks
Model is tested using two downstream tasks:

- Time series classification
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

Run classification code using ```python -m downstream_tasks.classification``` from project root

### Datasets
- Time series classification is done on the [UCR Dataset](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/) and the [p2s dataset](https://huggingface.co/datasets/AIML-TUDA/P2S) by AIML group of TU Darmstadt
  - To use the p2s dataset, Login using `huggingface-cli login` 
  - pip install -U "huggingface_hub[cli]" -> huggingface-cli login -> provide access token
- Portfolio management task is done on stock return time series of stocks in Nasdaq-100, downloaded from Yahoo Finance. 
We only use 78 of 100 stocks with available data in time span 01.2011 - 09.2024

### Benchmarks
The performance of this method is compared to SAX symbols and encodings using [VQShape](https://arxiv.org/pdf/2411.01006).
To use VQShape, first download a pretrained checkpoint [here](https://github.com/YunshiWen/VQShape/releases/tag/v0.1.0-cls),
extract the file and copy to `benchmarks/VQShape` folder.