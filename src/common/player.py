import numpy as np
import random
import time
import pygame as pg
from src.common.settings import PLAYER_RADIUS, PLAYER_COLOR, PLAYER_DEATH_COLOR, PLAYER_START_HEIGHT, PLAYER_START_POS, JUMP_FORCE, HEIGHT, BASE_HEIGHT, GRAVITY, MUTATION_SIZE, OBSTACLE_SPEED, MUTATION_CHANCE, PLAYER_JUMP_COOLDOWN, GAME_FPS, DECISION_THRESHOLD, FITNESS_WEIGHT_ALIVE, FITNESS_WEIGHT_KEYSCORE, WIDTH
from src.common.obstacle import Obstacle
from src.common.gate import Gate
from src.common.key import Key


class Player():
    def __init__(self, is_AI: bool = False) -> None:
        self.radius = PLAYER_RADIUS
        self.x = PLAYER_START_POS
        self.y = PLAYER_START_HEIGHT
        self.vy = 0
        self.jump_time = time.time()
        self.jump_cd = PLAYER_JUMP_COOLDOWN
        self.is_alive = True
        self.is_animating = False  # enabled by kill(), disabled by animation()
        self.init_time = time.time()
        self.time_alive = 0
        self.score = 0
        self.keyscore = 0
        self.has_key = False
        self.is_AI = is_AI  # NOTE: bit silly to define as new attribute
        # AI-only attributes:
        if self.is_AI:
            self.dy_bottom = 0
            self.dy_top = 0
            self.dx = 0
            self.dy_key = 0
            self.dx_key = 0
            self.weights_input = np.random.normal(0, scale=0.1, size=(7, 4))
            self.weights_hidden = np.random.normal(0, scale=0.1, size=(4, 1))

    def draw(self, screen) -> None:
        if self.is_alive:
            pg.draw.circle(screen, PLAYER_COLOR,
                           (self.x, self.y), PLAYER_RADIUS)
        else:
            self.animation(screen)

    def update(self, obstacle: Obstacle, key: Key, fps: int = GAME_FPS) -> None:
        if self.x >= obstacle.x + obstacle.width:
            self.has_key = False
        if self.is_alive:
            # If AI toggled, handle AI jumping
            if self.is_AI:
                self.NN_update(obstacle, key)
                if self.NN_jump():
                    self.jump(fps)
        # Handle ground collision and gravity
        if self.y >= HEIGHT - BASE_HEIGHT - self.radius and self.vy >= 0:
            self.y = HEIGHT - BASE_HEIGHT - self.radius
            self.vy = 0
        else:
            self.vy += GRAVITY
        # Update kinematics
        self.y += self.vy

    def jump(self, fps: int = GAME_FPS) -> None:
        if time.time() - self.jump_time >= self.jump_cd * 60 / fps:
            self.vy = JUMP_FORCE
            self.jump_time = time.time()

    def is_colliding(self, obstacle: Obstacle, gate: Gate) -> bool:
        # 1: Check if gate is open
        if not gate.is_open:
            dx = obstacle.x - self.x
            dy = 0
        # 2: If open, check if player has keey
        elif not self.has_key:
            dx = obstacle.x - self.x
            dy = 0
        # 3: If yes, gate will not collide
        else:
            dx = self.x - \
                max(obstacle.x, min(self.x, obstacle.x + obstacle.width))

            if obstacle.category == 'bottom':
                dy = self.y - \
                    max(obstacle.y, min(self.y, obstacle.y + obstacle.height))
            else:
                dy = self.y - \
                    max(obstacle.y - obstacle.height, min(self.y, obstacle.y))

        return (dx**2 + dy**2 <= self.radius**2) or (self.y - self.radius <= BASE_HEIGHT) or (self.y + self.radius >= HEIGHT - BASE_HEIGHT)

    def is_touching(self, key: Key) -> bool:
        dx = self.x - max(key.x, min(self.x, key.x + key.size))
        dy = self.y - max(key.y, min(self.y, key.y + key.size))
        return (dx**2 + dy**2 <= self.radius**2)

    def animation(self, screen):
        if self.is_animating and ((self.x + self.radius >= 0) or self.radius >= 0):
            self.x -= OBSTACLE_SPEED
            self.radius -= 1
            pg.draw.circle(screen, PLAYER_DEATH_COLOR,
                           (self.x, self.y), self.radius)
        else:
            self.is_animating = False

    def NN_update(self, obstacle: Obstacle, key: Key):
        """
        Updates player distances to ceiling and obstacles (AI player vision)
        """
        self.dx = obstacle.x - self.x
        if not self.has_key:
            self.dx_key = self.x - max(key.x, min(self.x, key.x + key.size))
            self.dy_key = self.y - max(key.y, min(self.y, key.y + key.size))
        else:
            self.dx_key = 0
            self.dy_key = 0
        # NOTE: Think more about the stuff below. Draw sketch
        if obstacle.category == 'bottom':
            self.dy_top = self.y - BASE_HEIGHT
            self.dy_bottom = obstacle.y - self.y
        else:
            self.dy_top = obstacle.y - self.y
            self.dy_bottom = HEIGHT - BASE_HEIGHT - self.y

    def NN_jump(self):
        genes = [self.y,
                 self.vy,
                 self.dx,
                 self.dy_bottom,
                 self.dy_top,
                 self.dx_key,
                 self.dy_key]
        hidden_layer_in = np.dot(genes, self.weights_input)
        hidden_layer_out = self.sigmoid(hidden_layer_in)
        output_layer_in = np.dot(hidden_layer_out, self.weights_hidden)
        prediction = self.sigmoid(output_layer_in)
        if prediction > DECISION_THRESHOLD:
            return True
        else:
            return False

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def kill(self):  # please don't
        self.is_alive = False
        self.is_animating = True
        self.time_alive = round(time.time() - self.init_time, 3)

    def mutate(self):
        for nw in range(len(self.weights_input)):
            for w in range(len(self.weights_input[nw])):
                if random.random() <= MUTATION_CHANCE:
                    self.weights_input[nw][w] += self.mutation_rate()

        for nw in range(len(self.weights_hidden)):
            for w in range(len(self.weights_hidden[nw])):
                if random.random() <= MUTATION_CHANCE:
                    self.weights_hidden[nw][w] += self.mutation_rate()

    def mutation_rate(self):
        # "dynamic" random "learning rate" to simulate mutation
        # NOTE: Consider if adding regularization possible
        # NOTE: Consider if adaptive step-size / learning rate is possible
        learning_rate = MUTATION_SIZE * np.random.uniform(-0.25, 0.25)
        return learning_rate

    def fitness(self):
        # Returns the normalized weighted sum of performance metrics
        # These include:
        #   Time alive (doesn't factor in FPS changes; normalization factor is an approximation)
        #       Exact normalization doesn't matter; deviations can be absorbed by the weight
        #   Keys collected
        # Player that collects the key is awarded more points
        norm_factor = WIDTH / OBSTACLE_SPEED / GAME_FPS
        return round((FITNESS_WEIGHT_ALIVE * self.time_alive / norm_factor + FITNESS_WEIGHT_KEYSCORE * self.keyscore), 3)
