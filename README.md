# Transformer-Time-Series-Data

 ## Experimenting with the transformer architecture for time series data.
 This model is based on the paper by Wu et al (2020) [1] a.t.m.. In cases where the paper does not specify what value was used for a specific configuration/hyperparameter, the values from Vaswani et al (2017) [2] or its PyTorch source code is used.
 
 This class assumes that input layers, positional encoding layers and linear mapping layers are separate from the encoder and decoder, i.e. implemented inside the present class and not inside the Encoder() and Decoder() classes.

    [1] Wu, N., Green, B., Ben, X., O'banion, S. (2020). 
    'Deep Transformer Models for Time Series Forecasting: 
    The Influenza Prevalence Case'. 
    arXiv:2001.08317 [cs, stat] [Preprint]. 
    Available at: http://arxiv.org/abs/2001.08317 (Accessed: 9 March 2022).

    [2] Vaswani, A. et al. (2017) 
    'Attention Is All You Need'.
    arXiv:1706.03762 [cs] [Preprint]. 
    Available at: http://arxiv.org/abs/1706.03762 (Accessed: 9 March 2022).
    
  
