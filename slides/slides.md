---
author: Tobenna P. Igwe
title: Deep Tensor Neural Network
date:
revealjs-url: '.'
css:
    - 'https://fonts.googleapis.com/css?family=Roboto+Slab:700'
theme:
    - black
---

# Introduction

## Outline

- Data
- Model
- Training
- Results
- Conclusion

## Original Paper

- K.T. Schütt. F. Arbabzadah. S. Chmiela, K.-R. Müller, A. Tkatchenko.
- Quantum-chemical insights from deep tensor neural networks.

## Prior Work

- [https://github.com/ptigwe/champs](https://github.com/ptigwe/champs)

# Data

## QM8

- Input
    - SMILE
        - `[H]C(=O)N([H])[H]`
- Four main targets for prediction
    - E1-CC2
    - E2-CC2
    - f1-CC2
    - f2-CC2

## How to handle SMILES

- Need to convert from SMILES to Chemical structures
    <div>
    <span style="vertical-align: middle;">`[H]C(=O)N([H])[H]` </span>
    &rarr;
    <img src="mol.png" style="vertical-align: middle;" width="100px"/>
    </div>
- Retrieve graph structure
- Retrieve XYZ positions

## RDKit

::: incremental
- Library for cheminformatics
- Contains the tools needed for the task
    - `MolFromSmiles`
    - `GetAtomicNum`
    - `Is3D`
    - `Get3DDistanceMatrix`
    - `GetDistanceMatrix`
:::

## Issues with RDKit

::: incremental
- Molecules need to be embedded
    - Embedding is stochastic
        - As a form of data augmentation
    - This process is not always successful
        - Fails especially on larger molecules
- Could use `GetMoleculeBoundsMatrix`
    - Conservative estimates of the distance
:::

## External Dataset

Sourced from [MoleculeNet](http://moleculenet.ai/datasets-1/)

- Benchmark dataset
- Contains the molecules in SDF format
    - Positional information is available

## External Dataset

Sourced from [MoleculeNet](http://moleculenet.ai/datasets-1/)

- Hydrogen
- Carbon
- Nitrogen
- Oxygen
- Fluorine

## External Dataset

Sourced from [MoleculeNet](http://moleculenet.ai/datasets-1/)

- Maximum of 26 atoms in a molecule
- 4 molecules are not 3D
- 32 molecules are not connected graphs
    - All 3D molecules

# Model

## Network Architecture

### Input

- $Z \in \mathbb{Z}^N$
    - A vector of nuclear charges
- $D \in \mathbb{R}^{N \times N}$
    - Matrix of atomic distances

where $N$ is the number of atoms in a molecule

## Network Architecture

![](network.png)

## Data Transformation

Gaussian Basis Expansion

$d_{ij} = \left[ exp\left(-\frac{(D_{ij} - (\mu_{min} + k \Delta\mu))^2}{2 \sigma^2} \right)\right]_{0 \le k \le \mu_{max} / \Delta \mu}$

```python
def gaussian_expansion(D, mu_min=-1, delta_mu=0.2, mu_max=10, sigma=0.2):
    mu = np.arange(mu_min, mu_max + delta_mu, delta_mu)
    diff = D[:,:,np.newaxis] - mu[np.newaxis, np.newaxis, :]
    return np.exp(-diff ** 2 / (2 * sigma))
```

## Embedding of Nuclear Charges

$c_i^{(0)} = c_{Z_i} \in R^B$

```python
self.C_embed = nn.Embedding(num_atoms, basis)
...
self.C_embed(Z_i)
```

## Interaction Module

$v_{ij} = tanh \left[W^{fc} \left(
(W^{cf} c_j + b^{f_1}) \cdot
(W^{df} \hat{d}_{ij} + b^{f_2}) 
 \right) \right]$

## Interaction Module

$v_{ij} = tanh \left[W^{fc} \left(
(W^{cf} c_j + b^{f_1}) \cdot
A_{ij} 
 \right) \right]$

where

$A_{ij} = W^{df} \hat{d}_{ij} + b^{f_2}$

```python
self.df = nn.Linear(num_gauss, basis)
...
A = self.df(D)
```

## Interaction Module

$v_{i} = tanh \left[W^{fc} \left(
(W^{cf} C + b^{f_1}) \cdot
A_{i} 
\right) \right]$

```python
self.cf = nn.Linear(basis, hidden)
self.fc = nn.Linear(hidden, basis, False)
...
X = self.cf(C)
X = X.unsqueeze(-2) * A
X = torch.tanh(self.fc(X))
```

## Coefficient Update

$$\begin{align}
c_i^{(t + 1)} & = c_i^{(t)} + \sum_{j \ne i}v_{ij}\\
             & = c_i^{(t)} + \sum m(i) * v_{i}
\end{align}$$

$m(i) = (1, \cdots, 1, 0, 1, \cdots, 1)$

```python
mask = utils.mask_2d(sizes, data.MAX_ATOMS)
(mask.unsqueeze(-1) * X).sum(-3)
```

## Coefficient Update
$$ mask = \left( \begin{matrix} 0 & 1 & 1 & 0 \\ 1 & 0 & 1 & 0 \\ 1 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{matrix} \right) $$

## Target Prediction

$o_i = tanh(W^{out_1}c_i^{(T)} + b^{out_1})$


$\hat{E}_{i} = W^{out_2}o_i + b^{out_2}$

```python
self.mlp = nn.Sequential(nn.Linear(basis, hidden),
                         nn.Tanh(),
                         nn.Linear(hidden, target))
```

## Output aggregation
$E = \sum_{i \in N}\hat{E}_{i}$

## Target Prediction

Two approaches for extending to multiple target values

- Single MLP
    - Single MLP with multiple outputs
- Multiple MLPs
    - Multiple networks connecting from $c_i^{(T)}$ to target

## Model Variation

```python
def target(basis, hidden, out):
    return nn.Sequential(nn.Linear(basis, hidden),
                         nn.Tanh(),
                         nn.Linear(hidden, 4)
```

Single Head

```python
mlp = target(30, 15, 4)
...
return mlp(C)
```

## Model Variation

```python
def target(basis, hidden, out):
    return nn.Sequential(nn.Linear(basis, hidden),
                         nn.Tanh(),
                         nn.Linear(hidden, 4)
```

Multi Head

```python
mlps = [target(30, 15, 1) for _ in range(4)]
...
return torch.cat([mlp(C) for mlp in mlps])
```

# Graph Neural Network

## Message Passing Neural Network

$\mathbf{x}_i^{(k)} = \gamma^{(k)} \left( \mathbf{x}_i^{(k-1)}, \sum_{j \in \mathcal{N}(i)} \, M^{(k)}_{ij} \right)$


$M^{(k)}_{ij} = \left(\mathbf{x}_i^{(k-1)}, \mathbf{x}_j^{(k-1)},\mathbf{e}_{j,i}\right)$

## Message

$M_{ij} = v_{ij}$

``` python

class InteractionBlockMLP(nn.Module):
    def __init__(self, basis, hidden):
        super().__init__()
        self.cf = nn.Linear(basis, hidden)
        self.df = nn.Linear(basis, hidden)
        self.fc = nn.Linear(hidden, basis, False)
```
    
    def forward(self, c, d):
        return torch.tanh(self.fc(self.cf(c) * self.df(d)))

## Message

``` python

class InteractionBlock(MessagePassing):
    def __init__(self, basis, hidden, **kwargs):
        super().__init__(**kwargs)
        self.inter_blk = InteractionBlockMLP(basis, hidden)
    
    def forward(self, x, edge_index, edge_attr):
        return x + self.propagate(edge_index, x=x,
                                  edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        return self.inter_blk(x_j, edge_attr)
```

## DTNN Network

``` python
class DTNN(nn.Module):
    def __init__(self, basis, hidden, T=3, **kwargs):
        super().__init__(**kwargs)
        self.embed = nn.Embedding(10, basis)
        self.inter_blk = InteractionBlock(basis, basis)
        self.readout_mlp = MLP(basis, hidden, 4)
        self.T = T

```

## DTNN Network

``` python        
    def forward(self, data):
        C = self.embed(data.Z)
        
        for _ in range(self.T):
            C = self.inter_blk(C, data.edge_index,
                               data.edge_attr)
        
        return global_add_pool(self.readout_mlp(C),
                               data.batch)
```

# Training

## Hyperparameters

- Used defaults as described in the paper
    - $\mu_{max} = 10$
    - $\mu_{min} = -1$
    - $\Delta{\mu} = \sigma = 0.2$
    - $B = 30$
    - $c_z \sim N(0, 1 / \sqrt{B})$

## Metrics and Evaluation

- MAE
- Train : Validation : Test = 8:1:1

Metrics and means of evaluation extracted from

- https://pubs.rsc.org/en/content/articlepdf/2018/sc/c7sc02664a
- https://jcheminf.biomedcentral.com/track/pdf/10.1186/s13321-019-0407-y
- https://arxiv.org/pdf/2008.12187.pdf

# Results

## Training

- 200 epochs
- Early stopping 
    - Patience 10
    - Average loss
- Adam Optimizer
    - Learning rate $10e^-4$

## WandB

[Weights and Biases](https://app.wandb.ai/ptigwe/DTNN/reports/DTNN--VmlldzoyMzAyMDQ?accessToken=5cnonc0b1m997oj2ez94ah73hc1cxggva99xp9agp91qwgbqfmvzs2ybv3rn2tyz)

# Any Questions

