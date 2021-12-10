## Extending VQ-VAE with Mutual Information




### Installation
``` 
conda install Pillow pytorch torchvision numpy tensorboardX six
```
### Training the VQ-VAE
1. Train baseline
`sbatch cluster-train-vqvae.job`
2. Train prior on baseline
Edit job file with --model-file /path/to/baseline/best.pt
`sbatch cluster-train-prior.job`
3. Train cpc-vqvae
`sbatch cluster-train-cpcvqvae.job`
4. Train prior on cpc
Edit job file with --model-file /path/to/cpc/best.pt
`sbatch cluster-train-prior.job`

### Credits

#### Original VQ-VAE implementation:
github: ritheshkumar95/pytorch-vqvae 

(See their original README [here]([README_OLD.md) ...)

#### CPC Module:
github: jefflai108/Contrastive-Predictive-Coding-PyTorch

#### Another Gated PixelCNN implementation:
This helped me extend Rithesh's sampler to 3-channel images:
https://github.com/anordertoreclaim/PixelCNN