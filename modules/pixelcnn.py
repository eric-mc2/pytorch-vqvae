import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger('pixelcnn-prior')

def weights_init(m):
    with torch.no_grad():
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            try:
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0)
            except AttributeError:
                print("Skipping initialization of ", classname)
    
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
        """x_v, x_h: (B, dim, D, D); h: (dim) """
        if self.mask_type == 'A':
            self.make_causal()

        logger.debug(f"h shape:: {h.shape}")
        h = self.class_cond_embedding(h) # h_e (dim, 2*dim)
        logger.debug(f"h_e shape: {h.shape}")
        h_vert = self.vert_stack(x_v)
        logger.debug(f"h_vert shape 1: {h_vert.shape}")
        h_vert = h_vert[:, :, :x_v.size(-1), :] # h_vert (dim, 2*dim, D, D)
        logger.debug(f"h_vert shape 2: {h_vert.shape}")
        out_v = self.gate(h_vert + h[:, :, None, None]) # out_v (dim, dim, D, D)
        
        logger.debug(f"x_h shape: {x_h.shape}")
        h_horiz = self.horiz_stack(x_h)
        logger.debug(f"h_horiz shape: {h_horiz.shape}")
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)] # h_horiz (dim, 2*dim, D, D)
        logger.debug(f"h_horiz2 shape: {h_horiz.shape}")
        v2h = self.vert_to_horiz(h_vert) # v2h (dim, 2*dim, D, D)

        logger.debug(f"v2h shape: {v2h.shape}")
        out = self.gate(v2h + h_horiz + h[:, :, None, None])
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)

        logger.debug(f"out_h shape: {out_h.shape}")
        logger.debug(f"out_v shape: {out_v.shape}")
        return out_v, out_h # out shape (dim, dim, D, D)


class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim=512, dim=64, n_layers=15, n_classes=10):
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
        """ x shape B x D x D, where D is downsampled H and W """
        logger.debug(f"x shape: {x.shape}")
        shp = x.size() + (-1, )
        
        x = self.embedding(x.view(-1)).view(shp)  # (B, D, D, dim)
        logger.debug(f"x shape: {x.shape}")
        x = x.permute(0, 3, 1, 2)  # (B, dim, D, D)

        x_v, x_h = (x, x)   # (B, dim, D, D)
        logger.debug(f"x_v, x_h shape: {x_v.shape}")
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)

        logger.debug(f"x_h shape: {x_h.shape}")
        output = self.output_conv(x_h)
        logger.debug(f"output shape: {output.shape}")
        return output
    
    def sideways(self, x, label):
        """ x shape B x D x D x C, where D is downsampled H and W """
        logger.debug(f"x shape: {x.shape}")
        shp = x.size()

        x_e = self.embedding(x.view(-1))
        logger.debug(f"x_e shape: {x_e.shape}")
        x = x_e.view((shp[0],shp[1],shp[2],shp[3]*x_e.shape[1]))  # (B, D, D, dim * C)
        logger.debug(f"x shape: {x.shape}")
        x = x.permute(0, 3, 1, 2)  # (B, C*dim, D, D)
        logger.debug(f"x shape: {x.shape}")

        x_v, x_h = (x, x)   # (B, C*dim, D, D)
        logger.debug(f"x_v, x_h shape: {x_v.shape}")
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)

        logger.debug(f"x_h shape: {x_h.shape}")
        output = self.output_conv(x_h)
        logger.debug(f"output shape: {output.shape}")
        return output
    
    def generate(self, 
        labels, 
        shape=(8, 8), 
        num_channels=3, 
        batch_size=64, 
        device='cuda'):

        # x are codes?
        x = torch.zeros((batch_size, *shape, num_channels), dtype=torch.int64).to(device)
        # samples = torch.zeros((batch_size, num_channels, *shape), dtype=torch.int64).to(device)
        labels = labels.to(device)

        logger.debug(f"generate: x shape: {x.shape}")
        logger.debug(f"generate: labels shape: {labels.shape}")
        for i in range(shape[0]):
            for j in range(shape[1]):
                for c in range(num_channels):
                    logits = self.sideways(x, labels) # (B, D, D, input_dim)
                    logits = logits.permute(0, 2, 3, 1).contiguous()
                    logger.debug(f"generate: logits shape: {logits.shape}")
                    probs = F.softmax(logits[:, i, j, :], -1)
                    logger.debug(f"generate: probs shape: {probs.shape}")
                     # shape [N,H,W,C], values [index of most likely pixel value]
                    sampled_levels = probs.multinomial(1).squeeze().data
                    logger.debug(f"generage: samp shape: {sampled_levels.shape}")
                     #shape [N,H,W,C]
                    # samples[:, c, i, j] = sampled_levels 
                    x.data[:, c, i, j].copy_(sampled_levels)
        return x

    def generate_bw(self, 
            labels, 
            shape=(8, 8), 
            batch_size=64, 
            device='cuda'):

            x = torch.zeros((batch_size, *shape), dtype=torch.int64).to(device)
            labels = labels.to(device)

            logger.debug(f"generate: x shape: {x.shape}")
            logger.debug(f"generate: labels shape: {labels.shape}")
            for i in range(shape[0]):
                for j in range(shape[1]):
                    logits = self.forward(x, labels) # (B, D, D)
                    logits = logits.permute(0, 2, 3, 1).contiguous()
                    logger.debug(f"generate: logits shape: {logits.shape}")
                    probs = F.softmax(logits[:, i, j, :], -1)
                    logger.debug(f"generate: probs shape: {probs.shape}")
                    # shape [N,H,W,K], values [index of most likely pixel value]
                    sampled_levels = probs.multinomial(1).squeeze().data
                    logger.debug(f"generage: samp shape: {sampled_levels.shape}")
                    #shape [N,H,W]
                    x.data[:, i, j].copy_(sampled_levels)
            return x[:, None, :, :]
