import torch.nn as nn
import logging

from modules.cpc import CPCModule
from modules.vqvae import VQVAEEncoder

logger = logging.getLogger('cpcvqvae-module')

class CPCVQVAE(nn.Module):
    def __init__(self, input_dim, dim, K=512, K_h=128, img_window=28*28, future_window=4*4):
        super().__init__()

        self.encoder = VQVAEEncoder(input_dim, dim ,K)
        self.cpc = CPCModule(K, K_h, img_window, future_window)
        
    def encode(self, x):
        return self.encoder.encode(x)
    
    def init_hidden(self, batch_size, K):
        return self.cpc.init_hidden(batch_size, K)

    def forward(self, x, hidden):
        z_q_x_st, z_e_x, z_q_x = self.encoder(x)
        accuracy, nce, z_e_x, z_q_x = self.cpc(z_q_x_st, z_e_x, z_q_x, hidden)
        return accuracy, nce, z_e_x, z_q_x
        