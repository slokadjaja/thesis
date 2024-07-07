## Symbolic Representation of Time Series Data using VAEs with Discrete Latent Variables

### Summary

We aim to learn symbolic representations of time series data in an unsupervised manner using 
variational autoencoders (VAEs) with categorical latent variables + Gumbel-softmax reparametrization

### Todos 

Priority:
- cat_kl_div: get n_latent, alphabet_size from vae class? -> less error prone
- see how to fix noisy loss, reconstruction
- enc, dec: 1d convolution, lstm
- implement 1 downstream task
- see if similar patches result in similar binary codes or
  - if similar binary codes corr. to similar patches
- play with hyperparams, activations, softmax temperature, number of bits
  - its ok if number of bits larger than patch length
- plot ts with real encodings
- clustering (dtw vs hamming)
- yaml file for specifying model architecture, hyperparams ([guide](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://rumn.medium.com/simplifying-machine-learning-workflow-with-yaml-files-e146cb3d481a%23:~:text%3DYAML%2520files%2520can%2520be%2520used,to%2520experiment%2520with%2520different%2520configurations.&ved=2ahUKEwja1disgpWHAxW8xQIHHcBPD6AQFnoECBIQAw&usg=AOvVaw0FvACmFdfonlEJTYirITe0))
- testing pipeline 

Nice to have

- additional "magnitude"+"shift" word? -> dependent on downstream tasks
- ensemble method to encode differnet attributes using differnt vaes
- generalization to other datasets
- Experiment management (wandb, mlflow)

### Questions

- normalize or split data first?
- is there a better way to split data into patches? (remove excess if length is not divisible by patch_len)