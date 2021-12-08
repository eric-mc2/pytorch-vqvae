# Train vqvae

PROJECT_ROOT="/Users/eric/Documents/Courses"
PROJECT_ROOT="$PROJECT_ROOT/UChicago Fall 2021/Deep Learning"
PROJECT_ROOT="$PROJECT_ROOT/final-project"
VQVAE_DIR="$PROJECT_ROOT/mi-vq-vae"
VQVAE_REPO="$VQVAE_DIR/pytorch-vqvae"

cd "$VQVAE_REPO"
# python -m pytest . -vv
# tensorboard --logdir logs

python vqvae.py --dataset mnist \
    --data-folder "$PROJECT_ROOT/data/mnist" \
    --output-folder models/vqvae-cpc \
    --batch-size 16 \
    --num-epochs 2 \
    --device cuda


# --device cuda \