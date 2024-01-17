# Transformer-Time-Series-Data

## Experimenting with the transformer architecture and other SOTA approaches for time series data.

Approaches can be found in the following papers. Code is based on respective the GitHub repositories:

### Autoformer:

see: https://arxiv.org/pdf/2106.13008.pdf
 
Autoformer redesigns the transformer architecture into a decomposition architecture with an auto-correlation mechanism. The model incorporates a Decomposition Layer to separate seasonality, trend-cycle components, and random fluctuation. Autoformer introduces an auto-correlation mechanism that replaces the standard self-attention used in the traditional transformer.

![autoformer_map](images/autoformer.png)
 
### TimesNet:

see: https://arxiv.org/pdf/2210.02186.pdf
 
This model addresses the phenomenon of multi-periodicity, such as daily and yearly variations for weather observations, and weekly and quarterly variations for electricity consumption, which overlap and interact. TimesNet converts 1-dimensional time series tensors into 2-dimensional tensors. We can reshape the 1D time series into a 2D tensor, where each column contains the time points within a period, and each row involves the time points at the same phase among different periods.

![timesnet_map](images/timesnet.png)
 
### LightTS:

see: https://arxiv.org/pdf/2207.01186.pdf
 
LightTS applies an MLP-based structure on top of two down-sampling strategies: 'interval sampling' and 'continuous sampling'. Down-sampling a time series often preserves the majority of its information and improves robustness. Continuous sampling involves converting a tensor of length T to T/C tensors of length C, where C represents consecutive points in the original tensor. Interval sampling involves selecting every C-th element to convert a tensor of length T to T/C tensors of length C. This method allows the model to capture both the local and global temporal patterns.

![lightts_map](images/LightTS.png)
 
### iTransformer:

see: https://arxiv.org/pdf/2310.06625.pdf
 
iTransformer applies a Transformer architecture without any modification to the basic components. iTransformer simply applies the attention and feed-forward network on the inverted input dimensions. Instead of embedding each temporal token, the time points of individual series are embedded into variate tokens which are utilized by the attention mechanism to capture multivariate correlations; meanwhile, the feed-forward network is applied for each variate token to learn nonlinear representations.

![itransformer_map](images/iTransformer.png)
  
