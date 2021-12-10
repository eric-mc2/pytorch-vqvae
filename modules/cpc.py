import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger('cpc-module')    

class CPCModule(nn.Module):
    def __init__(self, K=512, K_h=128, img_window=28*28, future_window=4*4):
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
        
        # mi/cpc modules
        logger.debug(f" GRU shape: 'input_features':{K}, 'hidden_features':{K_h}")
        self.gru = nn.GRU(K//2, K_h, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk  = nn.ModuleList([nn.Linear(K_h, K//2) for i in range(self.future_window_lin)])
        self.softmax  = nn.Softmax(dim=1)
        self.lsoftmax = nn.LogSoftmax(dim=1)

        for m in self.Wk:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            for layer_p in self.gru._all_weights:
                for p in layer_p:
                    if 'weight' in p:
                        nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')
    
    def init_hidden(self, batch_size, K):
        return torch.zeros(1, batch_size, K//4)

    def forward(self, z_q_x_st, z_e_x, z_q_x, hidden):
        """ z_e_x shape (B x K x D X D) """
        batch_size = z_e_x.shape[0]
        K = z_e_x.shape[1] 
        K_h = hidden.shape[-1]
        im_size_h = z_e_x.shape[2]
        im_size_w = z_e_x.shape[3]
        
        # permute z for gru
        # new z_q_x_.shape == B x D x D x K
        z_q_x_st_ = z_q_x_st.permute((0,2,3,1)) # (B,D,D,K)
        p_sample = torch.randint(self.max_sample_lin, size=(1,)).long() # randomly pick patches
                
        nce = torch.tensor(0.) # average over patches and batch
        # Encoded samples are for negative discriminator. Drawn from future.
        # encoded_samples.shape == F x B x K
        encoded_samples = torch.empty((self.future_window_lin, batch_size, K)).float()
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
        # output.shape == B x D*D x H
        # hidden.shape == 1 x B x H
        
        output, hidden = self.gru(forward_seq_lin, hidden)
        # c_t_.shape == B x H
        c_t_ = output[:, p_sample, :] 
        # c_t.shape == B x H
        c_t = c_t_.view(batch_size, K_h)
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
