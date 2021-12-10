import re
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from modules.vqvae import VQVAE
from datasets import download_datasets

from tensorboardX import SummaryWriter

logger = logging.getLogger('train-vqvae')

def train(data_loader, model, optimizer, args, writer):
    for images, _ in data_loader:
        images = images.to(args.device)

        optimizer.zero_grad()

        x_tilde, z_e_x, z_q_x = model(images)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, images)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq + args.beta * loss_commit
        loss.backward()

        # Logs
        writer.add_scalar('loss/train/reconstruction', loss_recons.item(), args.steps)
        writer.add_scalar('loss/train/quantization', loss_vq.item(), args.steps)

        optimizer.step()
        args.steps += 1


def test(data_loader, model, args, writer):
    with torch.no_grad():
        loss_recons, loss_vq = 0., 0.
        for images, _ in data_loader:
            images = images.to(args.device)
            x_tilde, z_e_x, z_q_x = model(images)
            loss_recons += F.mse_loss(x_tilde, images)
            loss_vq += F.mse_loss(z_q_x, z_e_x)

        loss_recons /= len(data_loader)
        loss_vq /= len(data_loader)

    # Logs
    writer.add_scalar('loss/test/reconstruction', loss_recons.item(), args.steps)
    writer.add_scalar('loss/test/quantization', loss_vq.item(), args.steps)

    return loss_recons.item(), loss_vq.item()


def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, _, _ = model(images)
    return x_tilde


def main(args):
    writer = SummaryWriter('./logs/{0}'.format(args.run_name))
    checkpoint_dir = './models/{0}'.format(args.run_name)

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
        batch_size=args.batch_size, shuffle=True)

    # Fixed images for Tensorboard
    fixed_images, _ = next(iter(test_loader))

    model = VQVAE(num_channels, args.hidden_size, args.k).to(args.device)

    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    checkpoint_re = re.compile(f'model_([0-9]+)')
    checkpoint_files = [os.path.basename(f) for f in os.listdir(checkpoint_dir)]
    checkpoint_matches = [checkpoint_re.search(f) for f in checkpoint_files]
    checkpoint_epochs = [int(m.group(1)) for m in checkpoint_matches if m]
    if checkpoint_epochs:
        last_saved_epoch = max(checkpoint_epochs)
        last_checkpoint = torch.load(f'{checkpoint_dir}/model_{last_saved_epoch}.pt')
        model.load_state_dict(last_checkpoint['model_state_dict'])
        optimizer.load_state_dict(last_checkpoint['optimizer_state_dict'])
        start_epoch = last_checkpoint['epoch']
    else:
        start_epoch = 0

    # Generate the samples first once
    reconstruction = generate_samples(fixed_images, model, model, args)
    grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('reconstruction', grid, 0)

    best_loss = np.Inf
    for epoch in range(start_epoch, args.num_epochs):
        logger.info('Epoch: {0}/{1}'.format(epoch, args.num_epochs))
        
        train(train_loader, model, optimizer, args, writer)
        loss_recons, _ = test(valid_loader, model, args, writer)
        
        reconstruction = generate_samples(fixed_images, model, model, args)
        grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
        writer.add_image('reconstruction', grid, epoch + 1)

        # Save best
        if (epoch == 0) or (loss_recons < best_loss):
            best_loss = loss_recons
            with open(f'{checkpoint_dir}/best.pt', 'wb') as f:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': loss_recons,
                }, f)
        
        # Save checkpoint
        if (epoch == 0) or (epoch % 5 == 0):
            with open(f'{checkpoint_dir}/model_{epoch + 1}.pt', 'wb') as f:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_recons,
                }, f)

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE')

    # General
    parser.add_argument('--data-folder', type=str,
        help='name of the data folder')
    parser.add_argument('--dataset', type=str,
        help='name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet)')

    # Latent space
    parser.add_argument('--hidden-size', type=int, default=256,
        help='size of the latent vectors (default: 256)')
    parser.add_argument('--k', type=int, default=512,
        help='number of latent vectors (default: 512)')
    
    # Optimization
    parser.add_argument('--batch-size', type=int, default=128,
        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=100,
        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=2e-4,
        help='learning rate for Adam optimizer (default: 2e-4)')
    parser.add_argument('--beta', type=float, default=1.0,
        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')

    # Miscellaneous
    parser.add_argument('--run-name', type=str, default='vqvae',
        help='name of the output folder (default: vqvae)')
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
    
    # Create logs, models, outputs folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./models'):
        os.makedirs('./models')
    if not os.path.exists('./models/{0}'.format(args.run_name)):
        os.makedirs('./models/{0}'.format(args.run_name))
    
    args.steps = 0

    main(args)
