CPC:
    * Run cpc model (RUNNING old & new scheduler)
    * Run decoder on whichever is better
    * Get pictures
    * Run prior
    * Get pictures

Baseline:
    * Run basic VQ-VAE model (RUNNING)
    * Get decoded pictures
    * Run prior on this model.
    * Get generated pictures
    
Hedge:
    * Code kernel kmeans or EM + GMM on new branch on vasic vqvae
    * Run it
    * Get decoded pictures
    * Run prior
    * Get generated pictures

