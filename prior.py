import numpy as np
import torch
import torch.nn.functional as F
import json
import re
import logging
from torchvision.utils import make_grid

from modules import VectorQuantizedVAE, GatedPixelCNN
from datasets import download_datasets

from tensorboardX import SummaryWriter

logger = logging.getLogger('pixelcnn-prior')


def train(data_loader, model, prior, optimizer, args, writer):
    for images, labels in data_loader:
        with torch.no_grad():
            images = images.to(args.device)
            latents = model.encode(images)
            latents = latents.detach()

        labels = labels.to(args.device)
        logits = prior(latents, labels)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        optimizer.zero_grad()
        loss = F.cross_entropy(logits.view(-1, args.k),
                               latents.view(-1))
        loss.backward()

        # Logs
        writer.add_scalar('loss/train', loss.item(), args.steps)

        optimizer.step()
        args.steps += 1

def test(data_loader, model, prior, args, writer):
    with torch.no_grad():
        loss = 0.
        for images, labels in data_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            latents = model.encode(images)
            latents = latents.detach()
            logits = prior(latents, labels)
            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss += F.cross_entropy(logits.view(-1, args.k),
                                    latents.view(-1))

        loss /= len(data_loader)

    # Logs
    writer.add_scalar('loss/valid', loss.item(), args.steps)

    return loss.item()

def generate_samples(prior, im_shape, batch_size, K, device):
    # XXX: label must be one batch long!!!
    label = torch.tensor([0,1]).expand(64, 2).contiguous().view(-1).long()
    x_tilde = prior.generate(label, im_shape, batch_size, device)
    images = x_tilde.cpu().data.float() / (K - 1)
    return images
    
def main(args):
    writer = SummaryWriter('./logs/{0}'.format(args.run_name))
    checkpoint_dir = './models/{0}-prior'.format(args.run_name)

    if not os.path.exists(args.model_file):
        parser.print_help()
        return 1

    logging.basicConfig(level=getattr(logging, args.logger_lvl.upper()))
    logger.info(f'Running on device: {args.device}')
    if args.device == 'cuda':
        logger.info(f'CUDA device count {torch.cuda.device_count()}')

    train_dataset, valid_dataset, test_dataset, num_channels, im_shape = download_datasets(args)    

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=16, shuffle=True)

    # Save the label encoder
    with open('./models/{0}-prior/labels.json'.format(args.run_name), 'w') as f:
        json.dump(train_dataset._label_encoder, f)

    # Fixed images for Tensorboard
    fixed_images, _ = next(iter(test_loader))
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)

    model = VectorQuantizedVAE(num_channels, args.hidden_size_vae, args.k,
            img_window=im_shape[0]*im_shape[1], future_window=args.num_future).to(args.device)
    with open(args.model_file, 'rb') as f:
        state_dict = torch.load(f)
        model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    n_classes = len(train_dataset._label_encoder)
    logger.debug(f"Making prior with {n_classes} classes")
    prior = GatedPixelCNN(args.k, args.hidden_size_prior,
        args.num_layers, n_classes=2)
    optimizer = torch.optim.Adam(prior.parameters(), lr=args.lr)

    checkpoint_re = re.compile(f'model_([0-9]+)')
    checkpoint_files = [os.path.basename(f) for f in os.listdir(checkpoint_dir)]
    checkpoint_matches = [checkpoint_re.search(f) for f in checkpoint_files]
    checkpoint_epochs = [int(m.group(1)) for m in checkpoint_matches if m]
    if checkpoint_epochs:
        last_saved_epoch = max(checkpoint_epochs)
        last_checkpoint = torch.load(f'{checkpoint_dir}/model_{last_saved_epoch}.pt')
        prior.load_state_dict(last_checkpoint['model_state_dict'])
        optimizer.load_state_dict(last_checkpoint['optimizer_state_dict'])
        start_epoch = last_checkpoint['epoch']
    else:
        start_epoch = 0

    prior.to(args.device)

    samples = generate_samples(prior, im_shape, args.batch_size ,args.k, args.device)
    grid = make_grid(samples, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('generated', grid, 0)

    best_loss = np.Inf
    for epoch in range(start_epoch, args.num_epochs):
        logger.info('Epoch: {0}/{1}'.format(epoch, args.num_epochs))
        train(train_loader, model, prior, optimizer, args, writer)
        # The validation loss is not properly computed since
        # the classes in the train and valid splits of Mini-Imagenet
        # do not overlap.
        loss = test(valid_loader, model, prior, args, writer)
        
        samples = generate_samples(prior, im_shape, args.batch_size ,args.k, args.device)
        grid = make_grid(samples, nrow=8, range=(-1, 1), normalize=True)
        writer.add_image('generated', grid, epoch + 1)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open(f'{checkpoint_dir}/best.pt', 'wb') as f:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': prior.state_dict(),
                    'loss': loss,
                }, f)

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='PixelCNN Prior for VQ-VAE')

    # General
    parser.add_argument('--data-folder', type=str,
        help='name of the data folder')
    parser.add_argument('--dataset', type=str,
        help='name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet)')
    parser.add_argument('--model-file', type=str, required=True,
        help='filename containing the model')
    parser.add_argument('--run-name', type=str, default='prior',
        help='name of the output folder (default: prior)')

    # Latent space
    parser.add_argument('--hidden-size-vae', type=int, default=256,
        help='size of the latent vectors (default: 256)')
    parser.add_argument('--hidden-size-prior', type=int, default=64,
        help='hidden size for the PixelCNN prior (default: 64)')
    parser.add_argument('--k', type=int, default=512,
        help='number of latent vectors (default: 512)')
    parser.add_argument('--num-layers', type=int, default=15,
        help='number of layers for the PixelCNN prior (default: 15)')
    parser.add_argument('--num-future', type=int, default=4*4,
        help='number of latent patches to predict (default: 16)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=128,
        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=100,
        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=3e-4,
        help='learning rate for Adam optimizer (default: 3e-4)')

    # Miscellaneous
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda, default: cpu)')
    parser.add_argument('--logger-lvl', type=str, default='WARNING',
        choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"],
        help='set the printf verbosity')

    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        args.device = torch.device(args.device)
    else:
        args.device = 'cpu'

    # Create logs and models folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./models'):
        os.makedirs('./models')
    if not os.path.exists('./models/{0}-prior'.format(args.run_name)):
        os.makedirs('./models/{0}-prior'.format(args.run_name))
    
    args.steps = 0

    main(args)
