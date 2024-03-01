import numpy as np
import time
import pygame as pg
from constants import PLAYER_RADIUS, PLAYER_COLOR, PLAYER_DEATH_COLOR, PLAYER_START_HEIGHT, PLAYER_START_POS, JUMP_FORCE, HEIGHT, BASE_HEIGHT, GRAVITY, MUTATION_SIZE, OBSTACLE_SPEED
from obstacle import Obstacle


class Player():
    def __init__(self, is_AI: bool = False) -> None:
        self.radius = PLAYER_RADIUS
        self.x = PLAYER_START_POS
        self.y = PLAYER_START_HEIGHT
        self.vy = 0
        self.is_jumping = False
        self.is_alive = True
        self.init_time = time.time()
        self.time_alive = 0
        self.is_AI = is_AI
        # AI-only attributes:
        if self.is_AI:
            self.dy_bottom = 0
            self.dy_top = 0
            self.dx = 0
            # NOTE: consider removing since it is directly correlated with self.vy
            self.jump_power = -10
            self.weights_input = np.random.normal(0, scale=0.1, size=(6, 3))
            self.weights_hidden = np.random.normal(0, scale=0.1, size=(3, 1))

    def draw(self, screen) -> None:  # NOTE: Possibly implement in main instead
        if self.is_alive:
            pg.draw.circle(screen, PLAYER_COLOR,
                           (self.x, self.y), PLAYER_RADIUS)
        else:
            self.animation(screen)

    def update(self, obstacle: Obstacle) -> None:
        if self.is_alive:
            # If AI toggled, handle AI jumping
            if self.is_AI:
                self.NN_update(obstacle)
                if self.NN_jump():
                    # NOTE: If this doesn't work, add a cooldown on jumping (as in game_logic.py)
                    self.jump()
        # Handle ground collision and gravity
        if self.y >= HEIGHT - BASE_HEIGHT - self.radius and self.vy >= 0:
            self.y = HEIGHT - BASE_HEIGHT - self.radius
            self.vy = 0
            self.is_jumping = False
        else:
            self.vy += GRAVITY
        # Update kinematics
        self.y += self.vy

    def jump(self) -> None:
        self.is_jumping = True
        if self.is_AI:
            self.vy = self.jump_power
        else:
            self.vy = JUMP_FORCE

    def is_colliding(self, obstacle: Obstacle):
        dx = PLAYER_START_POS - \
            max(obstacle.x, min(PLAYER_START_POS, obstacle.x + obstacle.width))
        if obstacle.category == 'bottom':
            dy = self.y - \
                max(obstacle.y, min(self.y, obstacle.y + obstacle.height))
        else:
            dy = self.y - \
                max(obstacle.y - obstacle.height, min(self.y, obstacle.y))
        dist = dx**2 + dy**2
        return (dist < self.radius**2) or (self.y - self.radius <= BASE_HEIGHT)

    def animation(self, screen):
        if (self.x + self.radius >= 0) or self.radius >= 0:
            self.x -= OBSTACLE_SPEED
            self.radius -= self.radius // 10
            pg.draw.circle(screen, PLAYER_DEATH_COLOR,
                           (self.x, self.y), self.radius)
        else:
            self.x = -self.radius

    def NN_update(self, obstacle: Obstacle):
        """
        Updates player distances to ceiling and obstacles (AI player vision)
        """
        self.dx = obstacle.x - self.x
        # NOTE: Think more about the stuff below. Draw sketch
        if obstacle.category == 'bottom':
            self.dy_top = self.y - BASE_HEIGHT
            self.dy_bottom = obstacle.y - self.y
        else:
            self.dy_top = obstacle.y - self.y
            self.dy_bottom = HEIGHT - self.y

    def NN_jump(self):
        bias = -0.5  # NOTE: Just -0.5 for now
        genes = [self.y, self.vy, self.dx,
                 self.dy_bottom, self.dy_top, self.jump_power]
        hidden_layer_in = np.dot(genes, self.weights_input)
        hidden_layer_out = self.sigmoid(hidden_layer_in)
        output_layer_in = np.dot(hidden_layer_out, self.weights_hidden)
        prediction = self.sigmoid(output_layer_in)
        if prediction + bias > 0:
            return True
        else:
            return False

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def kill(self):  # please don't
        self.is_alive = False
        self.time_alive = round(time.time() - self.init_time, 3)

    def fitness(self):
        return self.time_alive

    def mutate(self):
        for nw in range(len(self.weights_input)):
            for w in range(len(self.weights_input[nw])):
                self.weights_input[nw][w] += self.mutation_rate()

        for nw in range(len(self.weights_hidden)):
            for w in range(len(self.weights_hidden[nw])):
                self.weights_hidden[nw][w] += self.mutation_rate()

    def mutation_rate(self):
        # "dynamic" random "learning rate" to simulate mutation
        # NOTE: Consider if adding regularization possible
        learning_rate = MUTATION_SIZE * 0.005 * np.random.randint(-25, 25)
        return learning_rate


"""
NOTE:
Two of the NN features are probably redundant as they are just redefinitions of other features:
- self.dyc is just self.y - BASE_HEIGHT -- REMOVED
- self.jump_power is just self.vy 

TBD: Make player stop moving when collided
"""
