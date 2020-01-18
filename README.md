# INAS
Infinite Neural Architecture Search

Authors: Mian Akbar Shah, Giulia Lanzillotta, Sean Bergeron and Soley Hafthorsdottir


## Neural Architecture Search (NAS)
This project proposes an expansion of the work  the paper by Zoph and Le, 2017, *Neural Architecture Search with Reinforcement learning*. which can be found with the following link:
https://arxiv.org/abs/1611.01578

Neural Architecture Search (NAS) automates the process of designing neural network architectures. Designing high-quality neural networks requires specialized knowledge and extensive experience. Neural Architecture Search answers these requirements by automating the process of architecture engineering. Due to the work of Zoph and Le (2017), (see the link above) which framed NAS as a Reinforcement Learning problem, research on the topic has flourished. A wide variety of methods that tackled different issues have been published in quick succession. Within this expanding research area, our work focuses on the boundaries of the search space. More specifically, we challenge the design choices of Zoph and Le for the state and action spaces in their paper, which have been rarely investigated in recent years works. Our study is on the search for convolutional architectures, but we believe it can be applied to different architectures. Infinite Neural Architecture Search, INAS, obtains an infinite search space as a consequence of explicitly separating the action and state spaces. 

## The model

The model (see Figure below) consists in a pair of networks: a recurrent network (the controller), and a convolutional network (the child architecture).

![alt text](https://github.com/GiuliaLanzillotta/NAS/blob/Attention/NAS.jpg)

The training is organised in episodes and each episode consists of multiple steps. 
Starting from a fixed initial state, the controller progressively edits the child network's configuration, specified by a string containing all modifiable hyperparameters. The CNN is trained and its accuracy on a validation set acts as the reward signal. At the end of the episode we can compute the policy gradient to update the controller. As a result, in the next iteration, the controller will give higher probabilities to sequence of actions that receive high accuracies. In other words, the controller will learn to improve its search over time.

The following figure summarises the essential characteristics of every controller.

![alt text](https://github.com/GiuliaLanzillotta/NAS/blob/Attention/NAScontroller.png)
## Repo organisation

The implementation consists of four different modules.
The organisation of modules is as follows:
- the `train.py` file is responsible for managing the main loop during 
which the controller and child architectures are trained;
- the `controller.py` file is responsible for defining the architecture
of the controller and implementing the update function , which is
based on REINFORCE, a policy gradient method;
- the `cnn.py` file is responsible for managing the child architecture,
building the new state from the old state and an action and keeping
track of the states as the training progresses;
- the `conv_net.py` file is the actual child architecture and exposes its
training and testing functions.

The repository is organised over different branches: each branch corresponds 
a conducted experiment. Mainly, branches differ in the implementation
of the controller module.

## Running the experiments
The following packages are required to run the code:
- pytorch (1.3 or later)
- pandas
- numpy
- torchvision (0.5)
### Train the controller 
 
```python
cd src
python train.py
```
To train the controller the CIFAR-10 dataset is used. So as soon as the program is launched it will start downloading locally the CIFAR-10 dataset. 
### Test the controller 
You can use the Attention based controller to create a CNN architecture with the test module. 
At the moment, only the CIFAR-10 and MNIST datasets are supported. To add a dataset, implement a new loading function in the test file, following the other loading functions guidelines. As soon as the program is launched it will start downloading locally the dataset. 
```python
cd src
python test.py --data [DATASET] --steps [STEPS]
```
usage : DATASET should be either CIFAR or MNIST. STEPS is the number of steps allowed for the controller search. 

## Controller
You can choose between the following controllers:

- `Stacked LSTMs` : made of a different LSTM for each hyperparameter cascaded in the order the hyperparameters appear in the state vector
- `Dynamic length LSTMs`: one LSTM for each hyperparameter, regardless of the given layer, stacked them together following the state vector ordering
- `Encoder-Decoder + Attention`: we add an Attention mechanism based on Bahdanau et al., 2016 on top of the decoder network. 

## Ongoing experiments (or future work)
We are still working on the implementation of the following: 

- `Graph-Convolutional controller` : in the style of Kipf and Welling, ICLR 2017. Each node corresponds to an hyperparameter and each connection encodes the implicit correlation between them, information which is currently ignored.
- `Penalty`: penalising the controller for illegal actions.
- `Exploration`: to integrate in REINFORCE, with the intention of speeding up the search.
Each of these experiments is implemented in a different branch.

