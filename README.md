# Deep Reinforcement Learning for Invasive Green Crab Management

European green crabs are prolific generalist predators known to cause considerable perturbations to the ecosystems they colonize. 
This has had a damaging economic effect, particularly for shellfisheries in the western american coast.
To counter this, there is a growing investment in catching these crabs and removing them from estuaries and bays they are in the process of colonizing.
A question that naturally appears here is *how to best allocate limited resources in a dynamically evolving colonizing event?*

Here we approach this question from the point of view of deep reinforcement learning (RL). 

## RL in environmental management

RL is a broad class of machine learning algorithms that are aimed at solving *adaptive control/management problems*: problems in which an agent takes action on an environment based on its observations of the enviornment. 
These algorithms have been used for, e.g., teaching computers how to [become prolific](https://www.science.org/doi/full/10.1126/science.aar6404?casa_token=Gq_WicEszrcAAAAA%3Ax2KMhvk4p7mdPuAgnA2MBW6xpzH63x6jWsSDJs9oGZtJ5geNZn_1BCHQ4Amk0ErXfEqqcjPss9FGpw) at board games like Chess and Go.
Here, the environment would be the position of the Chess pieces in the chess board, which the agent observes and uses that observation to decide its next move.
Other classic uses of RL are for playing [atari games](https://arxiv.org/abs/1312.5602) and for solving [physics-based](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.031086) optimal control problems.

This project is part of a wider research program in which we take a step to extend RL out of its usual ‘‘comfort zone:’’ using it for adaptive management problems that arise in environmental science.
Our focus on this project is to leverage the tools developed in RL to the problem of resource allocation for invasive green crab management.

## Why try RL out here?

Our project starts with the question: *How to best use the data we collect on green crab populations in order to make policy decisions?*
Traditional methods used in adaptive management problems like this one, e.g. optimal control theory and Bayesian decision theory, have a hard time using high dimensional observations in their decision processes.
However, the data collected in our problem is naturally high dimensional: each month a *catch per unit effort* is recorded, and several such observations are needed in order to have enough information to make an informed opinion---for example, a sequence observations are needed in order to distinguish whether a green crab population is has a firm foothold within a bay or estuary.

So that is our setting: how do we efficiently use these high dimensional observations? 

Here RL offers an edge over traditional methods: RL is naturally suited for problems with high dimensional observations.
(Think of chess, for instance, where an observation has 32 components---the location of each individual piece on the board.)

Our project aims to use this advantage in order to generate new, more responsive, quantitative policy rules that could complement traditional management approaches.

## Model used

The agent trains by interacting with an *integral projection model* which describes the population dynamics of green crabs together with the agent's observation process.
In short, the agent's actions correspond to numbers of cages used to trap crabs. 
This action hampers the growth of the crab population, and produces an observation of number of crabs caught per cage used.

Coming soon: a more detailed explanation of the integral projection model used!

# Installation

1. `git clone https://github.com/boettiger-lab/rl4greencrab.git`
2. `cd rl4greencrab`
3. `pip install .`

(Coming soon: publishing our tools on PyPI in order to provide an easier installation!)

# Training

We use YAML files to specify training instances, examples of these files can be found on `hyperpars`. 
Key components there are: specifying the RL algorithm (e.g. PPO or TQC), specifying the environment trained on (e.g. `GreenCrab-v2`) and the training duration (6M time steps in our examples).
To train using this framework, use the command `python scripts/train.py path/to/file.yml`.

