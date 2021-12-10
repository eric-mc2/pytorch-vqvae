## Extending VQ-VAE with Mutual Information




### Installation
``` 
conda install Pillow pytorch torchvision numpy tensorboardX six
```
### Training the VQ-VAE
1. To train on MNIST
```
PROJECT_ROOT=/path/to/somewhere/outside/of/repo
RUN_NAME="something-memorable"
python vqvae.py --dataset mnist \
    --data-folder "$PROJECT_ROOT/data/mnist" \
    --run-name "$RUN_NAME" \
    --model encoder \
    --batch-size 16 \
    --num-epochs 50 \
    --device cuda \
    --logger-lvl INFO
```

### Credits

#### Original VQ-VAE implementation:
github: ritheshkumar95/pytorch-vqvae 

(See their original README [here]([README_OLD.md) ...)

#### CPC Module:
github: jefflai108/Contrastive-Predictive-Coding-PyTorch

#### Another Gated PixelCNN implementation:
This helped me extend Rithesh's sampler to 3-channel images:
https://github.com/anordertoreclaim/PixelCNN