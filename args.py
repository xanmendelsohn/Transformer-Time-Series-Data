import torch
import numpy as np
import random

class Arguments:
    def __init__(self):
        self.fix_seed: int = 2021
        random.seed(self.fix_seed)
        torch.manual_seed(self.fix_seed)
        np.random.seed(self.fix_seed)

        # basic config
        self.task_name: str = 'long_term_forecast'
        self.is_training: int = 1
        self.model_id: str = 'test'
        self.model: str = 'Autoformer'

        # data loader
        self.data: str = 'custom'
        self.root_path: str = 'data/exchange_rate/'
        self.data_path: str = 'exchange_rate.csv'
        self.features: str = 'M'
        self.target: str = 'OT'
        self.freq: str = 'h'
        self.checkpoints: str = './checkpoints/'

        # forecasting task
        self.seq_len: int = 96
        self.label_len: int = 48
        self.pred_len: int = 96
        self.seasonal_patterns: str = 'Monthly'
        self.inverse: bool = True

        # imputation task
        self.mask_rate: float = 0.25

        # anomaly detection task
        self.anomaly_ratio: float = 0.25

        # model define
        self.top_k: int = 5
        self.num_kernels: int = 6
        self.enc_in: int = 7
        self.dec_in: int = 7
        self.c_out: int = 7
        self.d_model: int = 512
        self.n_heads: int = 8
        self.e_layers: int = 2
        self.d_layers: int = 1
        self.d_ff: int = 2048
        self.moving_avg: int = 25
        self.factor: int = 1
        self.distil: bool = True
        self.dropout: float = 0.1
        self.embed: str = 'timeF'
        self.activation: str = 'gelu'
        self.output_attention: bool = False
        self.channel_independence: int = 0

        # optimization
        self.num_workers: int = 10
        self.itr: int = 1
        self.train_epochs: int = 10
        self.batch_size: int = 32
        self.patience: int = 3
        self.learning_rate: float = 0.0001
        self.des: str = 'test'
        self.loss: str = 'MSE'
        self.lradj: str = 'type1'
        self.use_amp: bool = False

        # GPU
        self.use_gpu: bool = torch.cuda.is_available()
        self.gpu: int = 0
        self.use_multi_gpu: bool = False
        self.devices: str = '0,1,2,3'
        self.device_ids: List[int] = [int(id_) for id_ in self.devices.replace(' ', '').split(',')]

        # de-stationary projector params
        self.p_hidden_dims: List[int] = [128, 128]
        self.p_hidden_layers: int = 2