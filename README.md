# Context-Dependent Go/No-Go RNN Coursework

This repository contains a compact coursework project on context-dependent computation in recurrent neural networks. The task is a simplified three-context Go/No-Go problem designed to compare standard recurrent models with a low-rank RNN whose dynamics can be inspected mechanistically.


## Task

Each trial has five input channels:

```text
[stimulus, context_0, context_1, context_2, go_cue]
```

The stimulus is either `+1` or `-1`, and the context determines the correct output:

| Context | Rule |
| --- | --- |
| `context_0` | no-go: output `0`, regardless of stimulus |
| `context_1` | pro rule: output equals stimulus |
| `context_2` | anti rule: output equals negative stimulus |

The network is read out at the final go-cue timestep. Outputs are discretized into `-1`, `0`, and `+1` using fixed thresholds.

## Models

Three networks are included:

- `low_rank_rnn.py`: continuous-time low-rank RNN with recurrent matrix `J = U V^T`
- `full_rank_rnn.py`: continuous-time full-rank vanilla RNN
- `lstm_rnn.py`: LSTM baseline

All models are trained with the same supervised task objective. The low-rank model is the main object of analysis because its recurrent dynamics are constrained to a low-dimensional subspace.

## Main Training Script

Train a model with:

```bash
python3 train_context_gonogo.py --model low_rank --steps 10000 --hidden-size 256 --rank 1 --lr 0.0001
```

Other model choices:

```bash
python3 train_context_gonogo.py --model full_rank --steps 5000 --hidden-size 256 --lr 0.0001
python3 train_context_gonogo.py --model lstm --steps 5000 --hidden-size 256 --lr 0.0001
```

The script saves checkpoints, metrics, example predictions, and training plots under the selected output directory.

## Short-Context Variant

`train_context_gonogo_short_context.py` implements a harder variant where the context cue is presented briefly before the stimulus instead of being available throughout the trial.

```bash
python3 train_context_gonogo_short_context.py --model low_rank --steps 10000 --hidden-size 256 --rank 1 --lr 0.0001
```

## Analysis Notebook

The main analysis notebook is:

```text
compare_trained_networks.ipynb
```

It compares trained low-rank, full-rank, and LSTM models on:

- learning curves
- task accuracy
- trial output trajectories
- PCA visualization of hidden states
- low-rank input and readout geometry
- low-rank flow fields
- context-dependent local readout gain
- subpopulation gain and ablation analyses

## Mechanistic Question

The central mechanistic question is:

> How does context change the mapping from recurrent latent state to output?

For the rank-1 low-rank RNN, the local readout gain along the recurrent direction can be written as:

```text
dz / d kappa_rec0 = sum_i W_out,i * (m_i / ||m||) * (1 - tanh(h_i)^2)
```

This decomposition separates three factors:

- `W_out,i`: how unit `i` contributes to the output
- `m_i`: how unit `i` belongs to the recurrent output direction
- `1 - tanh(h_i)^2`: whether unit `i` is in a sensitive or saturated regime

The analysis suggests that context modulates the effective readout gain by shifting different readout-carrying subpopulations into different activity regimes, rather than simply adding a fixed output bias.

## Repository Structure

```text
homework_CN/
├── low_rank_rnn.py
├── full_rank_rnn.py
├── lstm_rnn.py
├── train_context_gonogo.py
├── train_context_gonogo_short_context.py
├── compare_trained_networks.ipynb
└── outputs/
```

## Requirements

The code uses:

- Python 3
- PyTorch
- NumPy
- pandas
- matplotlib
- seaborn
- tqdm
- Jupyter Notebook or JupyterLab

Install the core dependencies with:

```bash
pip install torch numpy pandas matplotlib seaborn tqdm jupyter
```

## Notes

This is a small coursework-scale project. The goal is not to build the most complex model, but to create a simple setting where context-dependent computation can be trained, visualized, and explained with low-rank dynamical systems tools.
