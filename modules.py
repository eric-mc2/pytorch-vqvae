import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import math
import logging

from functions import vq, vq_st

logger = logging.getLogger('vqvae-modules')

def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()

def weights_init(m):
    with torch.no_grad():
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            try:
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0)
            except AttributeError:
                print("Skipping initialization of ", classname)
    


class VAE(nn.Module):
    def __init__(self, input_dim, dim, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 5, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, z_dim * 2, 3, 1, 0),
            nn.BatchNorm2d(z_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, dim, 3, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 5, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)

        q_z_x = Normal(mu, logvar.mul(.5).exp())
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()

        x_tilde = self.decoder(q_z_x.rsample())
        return x_tilde, kl_div


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

class VectorQuantizedVAEDecoder(nn.Module):
    def __init__(self, input_dim, dim, codebook):
        super().__init__()

        self.codebook = codebook.copy()
        
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


class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, K=512, K_h=128, img_window=28*28, future_window=4*4):
        super().__init__()

        self.img_window_lin = img_window
        self.downsampling_factor = 16 # == x.shape / encoder(x).shape
        self.img_window_lin_down = self.img_window_lin // self.downsampling_factor
        self.future_window_lin = future_window
        self.future_window = math.ceil(math.sqrt(self.future_window_lin))
        self.max_sample_lin = self.img_window_lin_down - self.future_window_lin

        if self.max_sample_lin <= 0:
            logger.warning("Max prediction index <= 0!")
            self.max_sample_lin = 1
        self.max_sample = math.ceil(math.sqrt(self.max_sample_lin))
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.codebook = VQEmbedding(K, dim)

        # mi/cpc modules
        logger.debug(f" GRU shape: 'input_features':{K}, 'hidden_features':{K_h}")
        self.gru = nn.GRU(K//2, K_h, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk  = nn.ModuleList([nn.Linear(K_h, K//2) for i in range(self.future_window_lin)])
        self.softmax  = nn.Softmax(dim=1)
        self.lsoftmax = nn.LogSoftmax(dim=1)

        # init weights
        self.apply(weights_init)
        # Initialization from jefflai108. Takes 4m ...
        for m in self.Wk:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')
    
    def init_hidden(self, batch_size, K):
        return torch.zeros(1, batch_size, K//4)

    def encode(self, x):
        # THIS IS PROBABLY WRONG BECAUSE DOESN"T USE CPC RNN
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def forward(self, x, hidden):
        logger.debug(f" x shape {x.shape}")
        z_e_x = self.encoder(x) # B x K x D X D
        batch_size = x.shape[0]
        K = z_e_x.shape[1] 
        K_h = hidden.shape[-1]
        im_size_h = z_e_x.shape[2]
        im_size_w = z_e_x.shape[3]
        logger.debug(f" z_e_x shape {z_e_x.shape}")
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x) # B x K x D x D
        logger.debug(f" z_q_x shape {z_q_x.shape}")
        p_sample = torch.randint(self.max_sample_lin, size=(1,)).long() # randomly pick patches
        logger.debug(f" max samples: {self.max_sample_lin}")
        logger.debug(f" future window: {self.future_window}")
        # permute z for gru
        # new z_q_x_.shape == B x D x D x K
        z_q_x_st_, z_q_x_ = z_q_x_st.permute((0,2,3,1)), z_q_x.permute((0,2,3,1))
        logger.debug(f" z_q_x_ shape {z_q_x_.shape}")
        nce = torch.tensor(0.) # average over patches and batch
        # Encoded samples are for negative discriminator. Drawn from future.
        # encoded_samples.shape == F x B x K
        encoded_samples = torch.empty((self.future_window_lin, batch_size, K)).float()
        logger.debug(f" encoded_samples shape {encoded_samples.shape}")
        for i in torch.arange(1, self.future_window_lin+1):
            row = torch.div(p_sample+i, im_size_w, rounding_mode='floor')
            col = (p_sample+i)%im_size_w
            encoded_sample = z_q_x_st_[:, row, col, :]
            encoded_samples[i-1] = encoded_sample.view(batch_size, K)
        
        # Forward seq is input to GRU. Drawn from past.
        # forward_seq.shape == B x D*D x K
        row = torch.div(p_sample + 1, im_size_w, rounding_mode='floor')
        col = (p_sample + 1)%im_size_w
        forward_seq_lin = z_q_x_st_.reshape((batch_size, im_size_h*im_size_w, K))[:, :p_sample + 1, :]
        logger.debug(f" forward_seq shape {forward_seq_lin.shape}")
        logger.debug(f" hidden shape {hidden.shape}")
        # output.shape == B x D*D x H
        # hidden.shape == 1 x B x H
        # if forward_seq_lin.shape[0] == 0:
        #     logger.error(f"RNN input has 0 length: {forward_seq_lin.shape}")
        # elif hidden.shape[0] == 0:
        #     logger.error(f"Hidden RNN has 0 length: {hidden.shape}")
        output, hidden = self.gru(forward_seq_lin, hidden)
        logger.debug(f" output shape {output.shape}")
        logger.debug(f" hiddenoutput shape {hidden.shape}")
        logger.debug(f" psample {p_sample}")
        # c_t_.shape == B x H
        c_t_ = output[:, p_sample, :] 
        logger.debug(f" c_t_ shape {c_t_.shape}")
        # c_t.shape == B x H
        c_t = c_t_.view(batch_size, K_h)
        logger.debug(f" c_t shape {c_t.shape}")
        logger.debug(f" Wk0 shape {self.Wk[0]}")
        pred = torch.empty((self.future_window_lin, batch_size, K)).float()
        for i in torch.arange(0, self.future_window_lin):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in torch.arange(0, self.future_window_lin):
            total = torch.mm(encoded_samples[i], torch.transpose(pred[i], 0, 1))
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch_size))) # correct is a tensor
            nce = nce + torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor
        nce = -1. * nce / (batch_size * self.future_window_lin)
        accuracy = 1.*correct/batch_size
        return accuracy, nce, z_e_x, z_q_x


class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return F.tanh(x) * F.sigmoid(y)


class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual

        self.class_cond_embedding = nn.Embedding(
            n_classes, 2 * dim
        )

        kernel_shp = (kernel // 2 + 1, kernel)  # (ceil(n/2), n)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.horiz_resid = nn.Conv2d(dim, dim, 1)

        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    def forward(self, x_v, x_h, h):
        if self.mask_type == 'A':
            self.make_causal()

        h = self.class_cond_embedding(h)
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]
        out_v = self.gate(h_vert + h[:, :, None, None])

        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)

        out = self.gate(v2h + h_horiz + h[:, :, None, None])
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)

        return out_v, out_h


class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10):
        super().__init__()
        self.dim = dim

        # Create embedding layer to embed input
        self.embedding = nn.Embedding(input_dim, dim)

        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()

        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes)
            )

        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1)
        )

        self.apply(weights_init)

    def forward(self, x, label):
        shp = x.size() + (-1, )
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, W, W)

        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)

        return self.output_conv(x_h)

    def generate(self, label, shape=(8, 8), batch_size=64):
        param = next(self.parameters())
        x = torch.zeros(
            (batch_size, *shape),
            dtype=torch.int64, device=param.device
        )

        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x, label)
                probs = F.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )
        return x
