# Jumping Game using Genetic Algorithm in Python

## Introduction
In this project I will use genetic algorithms in a python framework to train a character to jump over simple obstacles while exposed to a gravitational force. I use the module [`pygad`](https://pygad.readthedocs.io/en/latest/) ([source code](https://github.com/ahmedfgad/GeneticAlgorithmPython)) for the genetic algorithm part. I use `pygame` to create the game environment and all the game logic.

## Features
The **character** actions and objective include:
- **Jumping**: An instantaneous vertical velocity component is added onto the vertical position of the character
- **Motion**: The character will have a (constant) forward motion, OR have a fixed x-location while the obstacle moves to the left
- **Gravity**: The character will be dragged back down by a constant acceleration
- **Obstacles**: The character should be able to jump over an obstacle, and will fail if touching the obstacle (game reset)

The **environment** will be written up in HTML canvas (or something similar). It will include:
- A starting point for the character to spawn
- An obstacle that will disable the character if touched
- A goal point that rewards the character for reaching it

## Implementation steps
The following is a roadmap of steps to be taken for creating the game:
1. Create the map environment and character
2. Create the game logic: 
   - Gravity 
   - Obstacle-character interaction
   - Character-ground interaction
   - Character jump and motion
3. Genetic algorithm:
   - Define **genome representation**: Parameters controlling the jump
   - Define **fitness function**: Parameters determining character performance
     - Distance traveled
     - Speed at which other side is reached
     - Reaching goal point / number of succesful jumps (if multiple    obstacles)
   - Evolve population of candidate solutions (characters) over multiple generations and optimize performance based on fitness function
   - Evaluate best performing candidates in game environment and deploy for  gameplay