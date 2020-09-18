# Deep Tensor Neural Network

Implementation of Deep Tensor Neural Network (DTNN) in PyTorch based on the paper 
[Quantum-Chemical Insights from Deep Tensor Neural Networks](https://arxiv.org/abs/1609.08259)
originally designed for prediction of energy of a molecule based on the QM9 dataset.
Slight modifications have been made on this implementation to accomodate for the multiple
target variables of the QM8 dataset.

Take note that there are 2 implementations. One which implements the network using primarily
vanilla Pytorch features, for this take a look at the `model.py`, `data.py`, and `train.py` files.
The second method implements the network using a message passing neural network, based of off
the `torch_geometry` library, look at the file `torch_geom.py`.

For an explanation of the inner workings and explanation of implementations do take a look at
the slides directory.
