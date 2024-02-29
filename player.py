import numpy as np
import time
import pygame as pg
from constants import PLAYER_RADIUS, PLAYER_COLOR, PLAYER_START_HEIGHT, PLAYER_START_POS, JUMP_FORCE, HEIGHT, BASE_HEIGHT, GRAVITY
from obstacle import Obstacle


class Player():
    def __init__(self, is_AI: bool = False) -> None:
        self.radius = PLAYER_RADIUS
        self.y = PLAYER_START_HEIGHT
        self.vy = 0
        self.is_jumping = False
        self.is_dead = False
        self.init_time = time.time()
        self.time_alive = 0
        self.is_AI = is_AI
        # AI-only attributes:
        if self.is_AI:
            self.dy = 0
            self.dx = 0
            self.dyc = 0
            self.jump_power = 0  # NOTE: consider removing since it is directly correlated with self.vy
            self.input_weights = np.random.normal(0, scale=0.1, size=(5, 3))
            self.hidden_weights = np.random.normal(0, scale=0.1, size=(3, 1))

    def draw(self, screen) -> None:  # NOTE: Possibly implement in main instead
        pg.draw.circle(screen, PLAYER_COLOR,
                       (PLAYER_START_POS, self.y), PLAYER_RADIUS)

    def update(self) -> None:
        # If AI toggled, handle AI jumping
        if self.is_AI and self.NN_jump():
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
            # NOTE: Should later be changed to constant JUMP_FORCE defined in main
            self.vy = JUMP_FORCE

    def is_colliding(self, obstacle: Obstacle):
        dx = PLAYER_START_POS - \
            max(obstacle.x, min(PLAYER_START_POS, obstacle.x + obstacle.width))
        dy = self.y - \
            max(obstacle.y, min(self.y, obstacle.y + obstacle.height))
        dist = dx**2 + dy**2
        return (dist < self.radius**2) or (self.y <= BASE_HEIGHT)

    def NN_jump(self):
        bias = -0.5  # NOTE: Just -0.5 for now
        genes = [self.y, self.vy, self.dy, self.dyc, self.jump_power]
        hidden_layer_in = np.dot(genes, self.input_weights)
        hidden_layer_out = self.sigmoid(hidden_layer_in)
        output_layer_in = np.dot(hidden_layer_out, self.hidden_weights)
        prediction = self.sigmoid(output_layer_in)

        if prediction + bias > 0:
            return True
        else:
            return False

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
