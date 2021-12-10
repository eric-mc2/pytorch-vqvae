import torch
import torch.nn as nn
import logging

from functions import vq, vq_st

logger = logging.getLogger('vqvae-module')

def weights_init(m):
    with torch.no_grad():
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            try:
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0)
            except AttributeError:
                print("Skipping initialization of ", classname)
    

class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class VQVAEEncoder(nn.Module):
    def __init__(self, input_dim, dim, K=512):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.codebook = VQEmbedding(K, dim)

        # init weights
        self.apply(weights_init)

    def encode(self, x):
        # THIS IS PROBABLY WRONG BECAUSE DOESN"T USE CPC RNN
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def forward(self, x):
        z_e_x = self.encoder(x) # (B,K,D,D)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x) # (B,K,D,D)
        return z_q_x_st, z_e_x, z_q_x
        

class VQVAEDecoder(nn.Module):
    def __init__(self, input_dim, dim, codebook):
        super().__init__()

        self.codebook = codebook.requires_grad_(False)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, z_q_x):
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde


class VQVAE(nn.Module):
    def __init__(self, input_dim, dim, K=512):
        super().__init__()

        self.encoder = VQVAEEncoder(input_dim, dim ,K)
        self.decoder = VQVAEDecoder(input_dim, dim, self.encoder.codebook)

    def encode(self, x):
        return self.encoder.encode(x)
    
    def decode(self, latents):
        return self.decoder.decode(latents)

    def forward(self, x):
        z_q_x_st, z_e_x, z_q_x = self.encoder(x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x

