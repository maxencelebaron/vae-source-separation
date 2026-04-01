# VAE-Based Audio Source Separation

> MVA: Time Series course | ENS Paris-Saclay
>
> Authors: Pako Maxence TEKOU, Matéo Roinard--Chauvet

## Overview

This project investigates **audio source separation** using Variational Autoencoders (VAEs),
framed as a denoising problem. Given a mixture M = S1 + S2, the model is trained to reconstruct
the spectrogram of the target source S1 (speech) from the mixture, treating S2 (bird sounds) as interference.

We compare three latent-space regularization strategies:
- *VAE*: standard KL divergence regularization
- *β-VAE*: extended KL constraint for disentanglement
- *ITL-AE*: non-parametric divergences from Information-Theoretic Learning

Both supervised (trained on noisy mixtures) and unsupervised (trained on clean sources, applied
to mixtures) training strategies are explored. All code was written from scratch, as no reference
implementation was available.

## Dataset

- *Source 1 (target)* LibriSpeech: read speech recordings (male & female)
- *Source 2 (interference)* wave2vec2-vd-brid-sound-classification-dataset: bird sounds in noisy environments
- 400 signals total, 80/20 train/test split, standardized to 4 seconds at 16 kHz
- Mixtures constructed at a fixed PSNR of 10 dB

## Method

1. **Preprocessing**: STFT (window 512, hop 128), log-magnitude spectrograms
2. **Model**: VAE / β-VAE / ITL-AE trained to reconstruct target spectrograms
3. **Post-processing**: VAE outputs used as oracle masks for a Wiener filter
4. **Reconstruction**: iSTFT applied to filtered magnitude + original mixture phase

## Evaluation

Models are evaluated using standard source separation metrics: **SDR**, **SIR** and **SAR**.

## Repository Structure
```
├── frommixtureseparator/     # Models trained directly on mixtures (supervised)
├── latentspaceseparator/     # Models trained on clean sources (unsupervised)
├── version_supervise/        # Supervised training experiments
├── version_non_supervise/    # Unsupervised training experiments
├── resultats/                # Output figures and plots
├── slides_time_series.pdf    # Presentation slides
├── An_Overview_of_Variational_Autoencoders_for_Source_Separation_Finance_and_Bio_Signal_Applications        # Reference article
├── rapport.pdf               # Project report
└── requirements.txt
```

## Requirements
```bash
pip install -r requirements.txt
```

## Reference
[1] A. Singh and T. Ogunfunmi, "An Overview of Variational Autoencoders for Source Separation,
Finance, and Bio-Signal Applications," *Entropy*, vol. 24, no. 1, p. 55, 2022.
https://doi.org/10.3390/e24010055
