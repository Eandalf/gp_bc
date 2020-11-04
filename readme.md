# Genetic Programming for a Binary Classification Problem
This piece of work was originally created for COMP3211 Fundamentals of Artificial Intelligence Assignment1 Problem3 at HKUST in 2020 Fall.

## Set Up
gp.py was based on numpy. Use `pip install --upgrade numpy` to install numpy if necessary.

## Design
The script demonstrates how to set up a training based on the genetic programming paradigm. To perform the binary classification, the perceptron structure was set up to have weights to be tuned to fit the data. Basically, the population of this GP, genetic programming, consists of 1000 instances of perceptrons (weight vectors).
* Generation 0
  * 1000 perceptrons. They are vectors with random values from 1 to -1.
* Fitness Function
  * The accuracy rate after running a program (an instance within a generation).
* Crossover
  1. The programs will be ordered from having the best accuracy to the worst one.
  2. They would be in pair in order, i.e., 1 with 2, 3 with 4, etc.
  3. In the pair, they will choose 4 different random positions to exchange their vectors (programs, representing the weights).
  4. Thus, they will produce 4 descendants while keeping themselves to the next generation.
* Copy
  1. The programs will be ordered from having the best accuracy to the worst one.
  2.  Given the size of the population is 1000, the first 20% with the best accuracy would be chosen.
* Mutation
  * Every child will have a possibility of 0.005 to have a mutation in a random position with the inversed signed value compared to the value before the mutation.
* Termination Condition
  * The evolution stops while the error rate of the best one is lower than 5% or reaching 1000 iterations.

## Notice
All rights reserved.