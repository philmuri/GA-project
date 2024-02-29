# Obstacle Jumping with NN Genetic Algorithm

## Introduction
In this project I will incorporate a genetic algorithm made from scratch to train a player to jump over obstacles. This is done in a python framework using `pygame` for the game development part. The project will include a single-player mode where the user can play the game and compare their performance to the AI.

## Features
The player actions include:
- **Jumping**: An instantaneous vertical velocity component is added onto the vertical position of the character.
- **Motion**: Player motion is simulated by the obstacles moving to the left while the player remains still.
- **Gravity**: A constant downward acceleration.
- **Obstacles**: Player collision with obstacles will lead to a loss and will define performance in terms of time survived before a collision.

## Workflow for the GA part
Here I have summarized the steps involved in the training phase:
1. **Initialization**: Define GA features/genes/NN inputs and fitness function and initialize weights
   - Features include distance to ceiling, horizontal distance to closest object, vertical distance to upper-left corner of closest object, jumping power, and the vertical position and velocity of the player. All these will collectively influence when the AI decides to jump.
   - Fitness function is defined according to best performance, i.e. the time alive. 
   - Weights are initialized according to a first good-guess of the features.
2. **Selection**: Evolve a player population through generations by evaluating time alive for each player. Each player's jump state (jump or don't jump) is determined by the output/prediction of the NN.
3. **Evolution**: Genetic operators (**crossover**, **mutation**) are applied to create the new population. The best solutions will be used to initialize the new population in three different ways (split equally):
   - **Standard crossover**: Breed and mutate two best solutions within generation.
   - **Cross-generation crossover**: Breed and mutate best solutions in current generation and best overall generation.
   - **Best solution cloning**: Clone the weights of the best solution in current generation.
   - (**Resilience bonus**: If player survives over many generations, give them a survival bonus)
   - **Population reset**: If the NN consistently performs worse than $x$ generations ago, re-initialize a fraction of the population with new random weights.
4. **Repeat**: Replace old population with new by updating player weights

**Note**: Mutations are achieved by randomly adjusting the learning rate (and randomly in positive or negative direction) before being added to the weights. Crossover is achieved by returning the averaged the weights of each partner. Cloning is achieved by copying the weights directly to the cloned player.

## Neural network structure
TBD

## Current issues
- Some Player class attributes are passed as arguments of type None - namely 'genes' and 'toughness'. This wouldn't be necessary if a proper reset() method is defined that re-initializes all attributes except the latter ones. My current implementation requires introducing new lists ('mutated_genes' and 'new_toughness'), which creates a bit of clutter that could be refactored.